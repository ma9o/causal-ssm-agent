"use client";

import { useEffect, useRef, useState } from "react";
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
  getSocketUrl,
  buildFilterMessage,
  onMessage,
}: {
  enabled: boolean;
  getSocketUrl: () => string;
  buildFilterMessage: () => unknown;
  onMessage: (message: TMessage, socket: ReconnectingWebSocket) => void;
}): PrefectSocketConnectionState {
  const [connectionState, setConnectionState] = useState<PrefectSocketConnectionState>("idle");
  const getSocketUrlRef = useRef(getSocketUrl);
  const buildFilterMessageRef = useRef(buildFilterMessage);
  const onMessageRef = useRef(onMessage);

  getSocketUrlRef.current = getSocketUrl;
  buildFilterMessageRef.current = buildFilterMessage;
  onMessageRef.current = onMessage;

  useEffect(() => {
    if (!enabled) {
      setConnectionState("idle");
      return;
    }

    let disposed = false;
    setConnectionState("connecting");

    const ws = new ReconnectingWebSocket(getSocketUrlRef.current(), ["prefect"], {
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
          ws.send(JSON.stringify(buildFilterMessageRef.current()));
          setConnectionState("streaming");
          return;
        }
        if (message.type === "auth_failure") {
          setConnectionState("error");
          return;
        }

        onMessageRef.current(message, ws);
      } catch {
        // Ignore parse errors
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
  }, [enabled]);

  return connectionState;
}
