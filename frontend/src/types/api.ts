export type MessageRole = "user" | "assistant" | "system";

export interface Message {
    id: string;
    role: MessageRole;
    content: string;
    timestamp: string;
    metadata?: Record<string, unknown>;
}

export interface ChatRequest {
    user_id: string;
    message: string;
    session_id?: string;
    metadata?: Record<string, unknown>;
}

export interface ChatResponse {
    session_id: string;
    message: string;
    timestamp: string;
    metadata?: Record<string, unknown>;
}

export interface SessionInfo {
    session_id: string;
    user_id: string;
    message_count: number;
    created_at: string;
    updated_at: string;
    is_active: boolean;
}

export interface UserSessionsResponse {
    user_id: string;
    session_count: number;
    sessions: SessionInfo[];
}

export interface SessionHistoryResponse {
    session_id: string;
    messages: Message[];
    total_messages: number;
}
