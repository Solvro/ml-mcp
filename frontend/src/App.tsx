import { useCallback } from "react";
import { Sidebar } from "./components/Sidebar/Sidebar";
import { ChatWindow } from "./components/Chat/ChatWindow";
import { EmptyState } from "./components/shared/EmptyState";
import { ErrorBanner } from "./components/shared/ErrorBanner";
import { useUserId } from "./hooks/useUserId";
import { useSessions } from "./hooks/useSessions";
import { useChat } from "./hooks/useChat";
import { useTheme } from "./hooks/useTheme";

export default function App() {
    const userId = useUserId();
    const { isDark, toggle } = useTheme();
    const {
        sessions,
        activeSessionId,
        setActiveSessionId,
        createNewSession,
        deleteSession,
        refresh,
        loading: sessionsLoading,
    } = useSessions(userId);

    const handleSessionCreated = useCallback(
        (sessionId: string) => {
            setActiveSessionId(sessionId);
        },
        [setActiveSessionId],
    );

    const { messages, isLoading, error, sendMessage } = useChat(
        userId,
        activeSessionId,
        handleSessionCreated,
        refresh,
    );

    async function handleSend(text: string) {
        await sendMessage(text);
    }

    const showChat = activeSessionId !== null || messages.length > 0;

    return (
        <div className="flex h-screen bg-white dark:bg-stone-900 text-stone-900 dark:text-stone-100 overflow-hidden">
            <Sidebar
                sessions={sessions}
                activeSessionId={activeSessionId}
                onNewChat={createNewSession}
                onSelectSession={setActiveSessionId}
                onDeleteSession={deleteSession}
                loading={sessionsLoading}
                isDark={isDark}
                onToggleTheme={toggle}
            />

            <main className="flex flex-col flex-1 overflow-hidden">
                {error && <ErrorBanner message={error} />}

                {showChat ? (
                    <ChatWindow messages={messages} isLoading={isLoading} onSend={handleSend} />
                ) : (
                    <EmptyState onSend={handleSend} />
                )}
            </main>
        </div>
    );
}
