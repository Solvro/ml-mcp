import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { Message } from "../types/api";

interface UseChatResult {
    messages: Message[];
    isLoading: boolean;
    error: string | null;
    sendMessage: (text: string) => Promise<string | null>;
}

export function useChat(
    userId: string,
    activeSessionId: string | null,
    onSessionCreated: (sessionId: string) => void,
    onMessagesChanged: () => void,
): UseChatResult {
    const [messages, setMessages] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!activeSessionId) {
            setMessages([]);
            return;
        }

        let cancelled = false;

        async function loadHistory() {
            if (!activeSessionId) return;
            try {
                const data = await api.getSessionHistory(activeSessionId);
                if (!cancelled) {
                    setMessages(data.messages);
                    setError(null);
                }
            } catch (err) {
                if (!cancelled) {
                    setError(err instanceof Error ? err.message : "Błąd pobierania historii");
                }
            }
        }

        void loadHistory();
        return () => {
            cancelled = true;
        };
    }, [activeSessionId]);

    const sendMessage = useCallback(
        async (text: string): Promise<string | null> => {
            const userMessage: Message = {
                id: crypto.randomUUID(),
                role: "user",
                content: text,
                timestamp: new Date().toISOString(),
            };

            setMessages((prev) => [...prev, userMessage]);
            setIsLoading(true);
            setError(null);

            try {
                const response = await api.chat({
                    user_id: userId,
                    message: text,
                    session_id: activeSessionId ?? undefined,
                });

                const assistantMessage: Message = {
                    id: crypto.randomUUID(),
                    role: "assistant",
                    content: response.message,
                    timestamp: response.timestamp,
                    metadata: response.metadata,
                };

                setMessages((prev) => [...prev, assistantMessage]);

                if (!activeSessionId) {
                    onSessionCreated(response.session_id);
                }

                onMessagesChanged();

                return response.session_id;
            } catch (err) {
                setError(err instanceof Error ? err.message : "Błąd wysyłania wiadomości");
                setMessages((prev) => prev.filter((m) => m.id !== userMessage.id));
                return null;
            } finally {
                setIsLoading(false);
            }
        },
        [userId, activeSessionId, onSessionCreated, onMessagesChanged],
    );

    return { messages, isLoading, error, sendMessage };
}
