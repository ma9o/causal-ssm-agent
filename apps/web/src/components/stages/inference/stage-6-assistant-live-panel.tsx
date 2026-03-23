"use client";

import { getUserApiKey } from "@/lib/auth";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";

import {
  Stage6AssistantCard,
  type Stage6AssistantStatus,
} from "./stage-6-assistant-panel";

export function Stage6AssistantLivePanel({ userId }: { userId: string }) {
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  const transport = useMemo(() => {
    const apiKey = getUserApiKey();
    return new DefaultChatTransport({
      api: "/api/stage-6-assistant",
      body: { userId },
      ...(apiKey ? { headers: { "x-openrouter-key": apiKey } } : {}),
    });
  }, [userId]);

  const { messages, sendMessage, status } = useChat({ transport });
  const isLoading = status === "streaming" || status === "submitted";

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  function submitPrompt(text: string) {
    if (!text.trim() || isLoading) return;
    void sendMessage({ text: text.trim() });
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const text = input.trim();
    if (!text || isLoading) return;
    setInput("");
    submitPrompt(text);
  }

  return (
    <Stage6AssistantCard
      messages={messages}
      status={status as Stage6AssistantStatus}
      input={input}
      onInputChange={setInput}
      onSubmit={handleSubmit}
      onExamplePrompt={submitPrompt}
      showExamplePrompts={messages.length === 0}
      messageFooter={<div ref={bottomRef} />}
    />
  );
}
