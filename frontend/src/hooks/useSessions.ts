import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { SessionInfo } from "../types/api";

const ACTIVE_SESSION_KEY = "solvro_active_session";

interface UseSessionsResult {
    sessions: SessionInfo[];
    activeSessionId: string | null;
    setActiveSessionId: (id: string | null) => void;
    createNewSession: () => void;
    deleteSession: (id: string) => Promise<void>;
    refresh: () => Promise<void>;
    loading: boolean;
    error: string | null;
}

export function useSessions(userId: string): UseSessionsResult {
    const [sessions, setSessions] = useState<SessionInfo[]>([]);
    const [activeSessionId, setActiveSessionIdState] = useState<string | null>(() =>
        localStorage.getItem(ACTIVE_SESSION_KEY),
    );
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const setActiveSessionId = useCallback((id: string | null) => {
        setActiveSessionIdState(id);
        if (id) {
            localStorage.setItem(ACTIVE_SESSION_KEY, id);
        } else {
            localStorage.removeItem(ACTIVE_SESSION_KEY);
        }
    }, []);

    const refresh = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await api.getUserSessions(userId, false);
            const sorted = [...data.sessions].sort(
                (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime(),
            );
            setSessions(sorted);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Błąd pobierania sesji");
        } finally {
            setLoading(false);
        }
    }, [userId]);

    useEffect(() => {
        void refresh();
    }, [refresh]);

    const createNewSession = useCallback(() => {
        setActiveSessionId(null);
    }, [setActiveSessionId]);

    const deleteSession = useCallback(
        async (id: string) => {
            await api.deleteSession(id);
            if (activeSessionId === id) {
                setActiveSessionId(null);
            }
            await refresh();
        },
        [activeSessionId, refresh, setActiveSessionId],
    );

    return {
        sessions,
        activeSessionId,
        setActiveSessionId,
        createNewSession,
        deleteSession,
        refresh,
        loading,
        error,
    };
}
