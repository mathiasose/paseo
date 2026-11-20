import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import path from "node:path";
import type {
  ProviderTimelineItem,
  ProviderToolCallDetail,
} from "@getpaseo/plugin/server/provider";
import type { Step } from "./wire.js";

export function toolItem(
  step: Step,
  id: string,
): Extract<ProviderTimelineItem, { type: "tool_call" }> {
  const detail = toolDetail(step);
  const name = step.tool_name || "Antigravity tool";
  const metadata = step.subagent_info ? { subagents: step.subagent_info.subagents } : undefined;
  const base = { type: "tool_call" as const, id, callId: id, name, detail, metadata };
  if (step.state === "ERROR") {
    const error =
      step.tool_info && step.tool_info.error
        ? step.tool_info.error.message
        : "Antigravity tool failed";
    return { ...base, status: "failed", error };
  }
  const status = step.state === "ACTIVE" ? "running" : "completed";
  return { ...base, status, error: null };
}

function toolDetail(step: Step): ProviderToolCallDetail {
  if (step.subagent_info) {
    const child = step.subagent_info.subagents[0];
    if (child)
      return {
        type: "sub_agent",
        subAgentType: child.type_name,
        description: child.initial_prompt,
        childSessionId: child.conversation_id === null ? undefined : child.conversation_id,
        log: child.log_uri,
      };
  }
  const info = step.tool_info;
  if (!info) return { type: "plain_text", label: step.tool_name, text: "" };
  const filePath =
    info.parameters.Path || info.parameters.AbsolutePath || info.parameters.TargetFile;
  if (info.name === "run_command" && typeof info.parameters.CommandLine === "string") {
    return { type: "shell", command: info.parameters.CommandLine, output: info.output };
  }
  if ((info.name === "read_file" || info.name === "view_file") && typeof filePath === "string")
    return { type: "read", filePath, content: info.output };
  if ((info.name === "write_file" || info.name === "write_to_file") && typeof filePath === "string")
    return {
      type: "write",
      filePath,
      ...(typeof info.parameters.Content === "string" ? { content: info.parameters.Content } : {}),
    };
  if (
    (info.name === "edit_file" ||
      info.name === "replace_file_content" ||
      info.name === "multi_replace_file_content") &&
    typeof filePath === "string"
  )
    return {
      type: "edit",
      filePath,
      ...(typeof info.parameters.Target === "string" ? { oldString: info.parameters.Target } : {}),
      ...(typeof info.parameters.Replacement === "string"
        ? { newString: info.parameters.Replacement }
        : {}),
    };
  // Native frames summarize writes; they do not carry the file contents or diff.
  return { type: "unknown", input: info.parameters, output: info.error || info.output };
}

export interface ReplayedTimelineEntry {
  item: ProviderTimelineItem;
  timestamp?: string;
}

interface TranscriptToolCall {
  name?: string;
  args?: Record<string, unknown>;
}

interface TranscriptStep {
  step_index?: number;
  source?: string;
  type?: string;
  status?: string;
  created_at?: string;
  content?: string;
  thinking?: string;
  tool_calls?: TranscriptToolCall[];
}

function resolveTranscriptPath(conversationId: string): string | null {
  const appDataDir =
    process.env.ANTIGRAVITY_CLI_HOME || path.join(homedir(), ".gemini", "antigravity-cli");
  const logDir = path.join(appDataDir, "brain", conversationId, ".system_generated", "logs");
  const fullPath = path.join(logDir, "transcript_full.jsonl");
  if (existsSync(fullPath)) return fullPath;
  const compactPath = path.join(logDir, "transcript.jsonl");
  if (existsSync(compactPath)) return compactPath;
  return null;
}

function extractUserRequestText(raw: string): string {
  const match = /<USER_REQUEST>\r?\n?([\s\S]*?)\r?\n?<\/USER_REQUEST>/.exec(raw);
  return (match ? match[1] : raw).trim();
}

function stripToolResultTimestamps(raw: string): string {
  return raw.replace(/^(?:Created At: [^\r\n]+\r?\nCompleted At: [^\r\n]+\r?\n)+/, "");
}

export async function readTranscriptHistory(
  conversationId: string,
): Promise<ReplayedTimelineEntry[]> {
  const transcriptPath = resolveTranscriptPath(conversationId);
  if (!transcriptPath) return [];
  let content: string;
  try {
    content = await readFile(transcriptPath, "utf8");
  } catch {
    return [];
  }

  const steps: TranscriptStep[] = [];
  for (const line of content.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      steps.push(JSON.parse(trimmed) as TranscriptStep);
    } catch {
      // Ignore malformed lines
    }
  }

  const entries: ReplayedTimelineEntry[] = [];
  for (let i = 0; i < steps.length; i++) {
    const step = steps[i];
    const stepIndex = typeof step.step_index === "number" ? step.step_index : i;
    const timestamp = typeof step.created_at === "string" ? step.created_at : undefined;

    if (step.type === "USER_INPUT" && typeof step.content === "string") {
      const text = extractUserRequestText(step.content);
      if (text) {
        const id = `history:user:${stepIndex}`;
        entries.push({
          timestamp,
          item: {
            type: "user_message",
            id,
            messageId: id,
            text,
          },
        });
      }
      continue;
    }

    if (step.type === "PLANNER_RESPONSE") {
      if (typeof step.thinking === "string" && step.thinking.trim()) {
        entries.push({
          timestamp,
          item: {
            type: "reasoning",
            id: `history:reasoning:${stepIndex}`,
            text: step.thinking.trim(),
          },
        });
      }

      if (typeof step.content === "string" && step.content.trim()) {
        const id = `history:assistant:${stepIndex}`;
        entries.push({
          timestamp,
          item: {
            type: "assistant_message",
            id,
            messageId: id,
            text: step.content,
          },
        });
      }

      if (Array.isArray(step.tool_calls) && step.tool_calls.length > 0) {
        // Pair each tool call with subsequent GENERIC result steps
        let resultCursor = i + 1;
        for (let tcIndex = 0; tcIndex < step.tool_calls.length; tcIndex++) {
          const tc = step.tool_calls[tcIndex];
          const toolName = typeof tc.name === "string" ? tc.name : "Antigravity tool";
          const parameters =
            tc.args && typeof tc.args === "object" && !Array.isArray(tc.args) ? tc.args : {};

          let resultStep: TranscriptStep | undefined;
          while (resultCursor < steps.length && steps[resultCursor].type === "GENERIC") {
            resultStep = steps[resultCursor];
            resultCursor++;
            break;
          }

          const rawOutput =
            resultStep && typeof resultStep.content === "string"
              ? stripToolResultTimestamps(resultStep.content)
              : "";
          const isError = resultStep?.status === "ERROR";
          const syntheticStep: Step = {
            conversation_id: conversationId,
            step_index: stepIndex,
            state: isError ? "ERROR" : "DONE",
            step_type: "tool",
            text_delta: "",
            tool_name: toolName,
            tool_info: {
              name: toolName,
              parameters: parameters as Record<string, unknown>,
              output: rawOutput,
              error: isError ? { type: "ToolError", message: rawOutput || "Tool failed" } : null,
            },
            subagent_info: null,
            usage: null,
          };
          const callId = `history:tool:${stepIndex}:${tcIndex}`;
          entries.push({
            timestamp:
              (resultStep && typeof resultStep.created_at === "string"
                ? resultStep.created_at
                : undefined) ?? timestamp,
            item: toolItem(syntheticStep, callId),
          });
        }
      }
    }
  }

  return entries;
}
