import { useState } from "react";
import type { SessionInfo } from "../../types/api";

interface SessionItemProps {
    session: SessionInfo;
    isActive: boolean;
    onClick: () => void;
    onDelete: () => void;
}

const dateFormatter = new Intl.DateTimeFormat("pl-PL", {
    dateStyle: "short",
    timeStyle: "short",
});

export function SessionItem({ session, isActive, onClick, onDelete }: SessionItemProps) {
    const [hovered, setHovered] = useState(false);

    function handleDelete(e: React.MouseEvent) {
        e.stopPropagation();
        onDelete();
    }

    return (
        <button
            onClick={onClick}
            onMouseEnter={() => setHovered(true)}
            onMouseLeave={() => setHovered(false)}
            className={`w-full text-left flex items-center justify-between gap-2 px-3 py-2.5 rounded-xl text-sm transition-colors ${
                isActive
                    ? "bg-topwr text-white"
                    : "text-stone-600 dark:text-stone-300 hover:bg-stone-100 dark:hover:bg-stone-800 hover:text-stone-800 dark:hover:text-stone-100"
            }`}
        >
            <div className="flex flex-col gap-0.5 overflow-hidden min-w-0">
                <span className="truncate text-xs font-medium">
                    {dateFormatter.format(new Date(session.updated_at))}
                </span>
                <span
                    className={`text-xs ${isActive ? "text-orange-100" : "text-stone-400 dark:text-stone-500"}`}
                >
                    {session.message_count}{" "}
                    {session.message_count === 1 ? "wiadomość" : "wiadomości"}
                </span>
            </div>
            {hovered && (
                <button
                    onClick={handleDelete}
                    className={`shrink-0 w-6 h-6 flex items-center justify-center rounded-lg transition-colors ${
                        isActive
                            ? "hover:bg-orange-600/30 text-orange-100"
                            : "hover:bg-red-100 text-stone-400 hover:text-red-500"
                    }`}
                    aria-label="Usuń sesję"
                >
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 20 20"
                        fill="currentColor"
                        className="w-3.5 h-3.5"
                    >
                        <path
                            fillRule="evenodd"
                            d="M8.75 1A2.75 2.75 0 006 3.75v.443c-.795.077-1.584.176-2.365.298a.75.75 0 10.23 1.482l.149-.022.841 10.518A2.75 2.75 0 007.596 19h4.807a2.75 2.75 0 002.742-2.53l.841-10.52.149.023a.75.75 0 00.23-1.482A41.03 41.03 0 0014 4.193V3.75A2.75 2.75 0 0011.25 1h-2.5zM10 4c.84 0 1.673.025 2.5.075V3.75c0-.69-.56-1.25-1.25-1.25h-2.5c-.69 0-1.25.56-1.25 1.25v.325C8.327 4.025 9.16 4 10 4zM8.58 7.72a.75.75 0 00-1.5.06l.3 7.5a.75.75 0 101.5-.06l-.3-7.5zm4.34.06a.75.75 0 10-1.5-.06l-.3 7.5a.75.75 0 101.5.06l.3-7.5z"
                            clipRule="evenodd"
                        />
                    </svg>
                </button>
            )}
        </button>
    );
}
