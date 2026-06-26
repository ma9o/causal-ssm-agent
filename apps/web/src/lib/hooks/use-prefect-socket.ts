"use client";

import { useEffect, useEffectEvent, useState } from "react";
import ReconnectingWebSocket from "reconnecting-websocket";

const MAX_RECONNECT_ATTEMPTS = 10;
const BASE_DELAY_MS = 1000;

export type PrefectSocketConnectionState =
  | "idle"
  | "connecting"
  | "authenticating"
  | "streaming"
  | "error";

interface PrefectSocketEnvelope {
  type?: string;
}

export function usePrefectSocketSubscription<TMessage extends PrefectSocketEnvelope>({
  enabled,
  subscriptionKey,
  getSocketUrl,
  buildFilterMessage,
  onSubscribed,
  onMessage,
}: {
  enabled: boolean;
  subscriptionKey: string;
  getSocketUrl: () => string;
  buildFilterMessage: () => unknown;
  onSubscribed?: (socket: ReconnectingWebSocket) => void;
  onMessage: (message: TMessage, socket: ReconnectingWebSocket) => void;
}): PrefectSocketConnectionState {
  const [connectionState, setConnectionState] = useState<PrefectSocketConnectionState>("idle");
  const resolveSocketUrl = useEffectEvent(() => getSocketUrl());
  const resolveFilterMessage = useEffectEvent(() => buildFilterMessage());
  const handleSubscribed = useEffectEvent((socket: ReconnectingWebSocket) => {
    onSubscribed?.(socket);
  });
  const handleMessage = useEffectEvent((message: TMessage, socket: ReconnectingWebSocket) => {
    onMessage(message, socket);
  });

  useEffect(() => {
    if (!enabled) {
      return;
    }

    let disposed = false;

    const ws = new ReconnectingWebSocket(resolveSocketUrl(), ["prefect"], {
      maxRetries: MAX_RECONNECT_ATTEMPTS,
      minReconnectionDelay: BASE_DELAY_MS,
      maxReconnectionDelay: BASE_DELAY_MS * 2 ** MAX_RECONNECT_ATTEMPTS,
      reconnectionDelayGrowFactor: 2,
    });

    ws.onopen = () => {
      if (disposed) {
        return;
      }
      setConnectionState("authenticating");
      ws.send(JSON.stringify({ type: "auth", token: null }));
    };

    ws.onmessage = (event: MessageEvent) => {
      if (disposed) {
        return;
      }

      try {
        const message = JSON.parse(event.data) as TMessage;
        if (message.type === "auth_success") {
          ws.send(JSON.stringify(resolveFilterMessage()));
          setConnectionState("streaming");
          handleSubscribed(ws);
          return;
        }
        if (message.type === "auth_failure") {
          setConnectionState("error");
          return;
        }

        handleMessage(message, ws);
      } catch (err) {
        console.warn("WebSocket message parse failed:", err);
      }
    };

    ws.onerror = () => {
      if (!disposed) {
        setConnectionState("error");
      }
    };

    ws.onclose = () => {
      if (!disposed) {
        setConnectionState("connecting");
      }
    };

    return () => {
      disposed = true;
      setConnectionState("idle");
      ws.close();
    };
  }, [enabled, subscriptionKey]);

  if (!enabled) {
    return "idle";
  }

  return connectionState === "idle" ? "connecting" : connectionState;
}
