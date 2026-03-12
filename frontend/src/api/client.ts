import type {
    ChatRequest,
    ChatResponse,
    SessionHistoryResponse,
    UserSessionsResponse,
} from "../types/api";

const BASE_URL = "";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
    const res = await fetch(`${BASE_URL}${path}`, {
        headers: { "Content-Type": "application/json", ...options?.headers },
        ...options,
    });

    if (!res.ok) {
        const text = await res.text().catch(() => res.statusText);
        throw new Error(`HTTP ${res.status}: ${text}`);
    }

    return res.json() as Promise<T>;
}

export const api = {
    chat(body: ChatRequest): Promise<ChatResponse> {
        return request<ChatResponse>("/api/chat", {
            method: "POST",
            body: JSON.stringify(body),
        });
    },

    getUserSessions(userId: string, activeOnly = false): Promise<UserSessionsResponse> {
        return request<UserSessionsResponse>(
            `/api/users/${userId}/sessions?active_only=${activeOnly}`,
        );
    },

    getSessionHistory(sessionId: string): Promise<SessionHistoryResponse> {
        return request<SessionHistoryResponse>(`/api/sessions/${sessionId}/history`);
    },

    deleteSession(sessionId: string): Promise<void> {
        return request<void>(`/api/sessions/${sessionId}`, { method: "DELETE" });
    },
};
