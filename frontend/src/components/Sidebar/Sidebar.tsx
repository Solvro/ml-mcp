import type { SessionInfo } from "../../types/api";
import { NewChatButton } from "./NewChatButton";
import { SessionItem } from "./SessionItem";

interface SidebarProps {
    sessions: SessionInfo[];
    activeSessionId: string | null;
    onNewChat: () => void;
    onSelectSession: (id: string) => void;
    onDeleteSession: (id: string) => void;
    loading: boolean;
    isDark: boolean;
    onToggleTheme: () => void;
}

export function Sidebar({
    sessions,
    activeSessionId,
    onNewChat,
    onSelectSession,
    onDeleteSession,
    loading,
    isDark,
    onToggleTheme,
}: SidebarProps) {
    return (
        <aside className="w-64 shrink-0 bg-stone-50 dark:bg-stone-900 flex flex-col h-full border-r border-stone-100 dark:border-stone-800">
            <div className="p-3 border-b border-stone-100 dark:border-stone-800">
                <div className="flex items-center justify-between px-2 py-1.5 mb-2">
                    <span className="text-sm font-bold tracking-tight text-stone-900 dark:text-stone-100">
                        PWr<span className="text-topwr">Chat</span>
                    </span>
                    <button
                        onClick={onToggleTheme}
                        className="w-7 h-7 flex items-center justify-center rounded-lg text-stone-400 hover:text-stone-600 dark:text-stone-500 dark:hover:text-stone-300 hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors"
                        aria-label={isDark ? "Tryb jasny" : "Tryb ciemny"}
                    >
                        {isDark ? (
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                viewBox="0 0 20 20"
                                fill="currentColor"
                                className="w-4 h-4"
                            >
                                <path d="M10 2a.75.75 0 01.75.75v1.5a.75.75 0 01-1.5 0v-1.5A.75.75 0 0110 2zM10 15a.75.75 0 01.75.75v1.5a.75.75 0 01-1.5 0v-1.5A.75.75 0 0110 15zM10 7a3 3 0 100 6 3 3 0 000-6zM15.657 5.404a.75.75 0 10-1.06-1.06l-1.061 1.06a.75.75 0 001.06 1.06l1.06-1.06zM6.464 14.596a.75.75 0 10-1.06-1.06l-1.06 1.06a.75.75 0 001.06 1.06l1.06-1.06zM18 10a.75.75 0 01-.75.75h-1.5a.75.75 0 010-1.5h1.5A.75.75 0 0118 10zM5 10a.75.75 0 01-.75.75h-1.5a.75.75 0 010-1.5h1.5A.75.75 0 015 10zM14.596 15.657a.75.75 0 001.06-1.06l-1.06-1.061a.75.75 0 10-1.06 1.06l1.06 1.06zM5.404 6.464a.75.75 0 001.06-1.06L5.403 4.343a.75.75 0 00-1.06 1.06l1.06 1.061z" />
                            </svg>
                        ) : (
                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                viewBox="0 0 20 20"
                                fill="currentColor"
                                className="w-4 h-4"
                            >
                                <path
                                    fillRule="evenodd"
                                    d="M7.455 2.004a.75.75 0 01.26.77 7 7 0 009.958 7.967.75.75 0 011.067.853A8.5 8.5 0 116.647 1.921a.75.75 0 01.808.083z"
                                    clipRule="evenodd"
                                />
                            </svg>
                        )}
                    </button>
                </div>
                <NewChatButton onClick={onNewChat} />
            </div>

            <div className="flex-1 overflow-y-auto p-3">
                {loading && sessions.length === 0 ? (
                    <div className="text-xs text-stone-400 dark:text-stone-500 text-center py-4">
                        Ładowanie...
                    </div>
                ) : sessions.length === 0 ? (
                    <div className="text-xs text-stone-400 dark:text-stone-500 text-center py-4">
                        Brak rozmów
                    </div>
                ) : (
                    <div className="flex flex-col gap-1">
                        <p className="text-xs text-stone-400 dark:text-stone-500 px-3 mb-1 uppercase tracking-wide">
                            Rozmowy
                        </p>
                        {sessions.map((s) => (
                            <SessionItem
                                key={s.session_id}
                                session={s}
                                isActive={s.session_id === activeSessionId}
                                onClick={() => onSelectSession(s.session_id)}
                                onDelete={() => onDeleteSession(s.session_id)}
                            />
                        ))}
                    </div>
                )}
            </div>

            <div className="p-3 border-t border-stone-100 dark:border-stone-800">
                <p className="text-xs text-stone-400 dark:text-stone-500 text-center">
                    Politechnika Wrocławska · ToPWR
                </p>
            </div>
        </aside>
    );
}
