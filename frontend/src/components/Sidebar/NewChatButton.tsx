interface NewChatButtonProps {
    onClick: () => void;
}

export function NewChatButton({ onClick }: NewChatButtonProps) {
    return (
        <button
            onClick={onClick}
            className="flex items-center gap-2 w-full px-3 py-2 rounded-xl text-sm text-stone-600 dark:text-stone-300 hover:bg-stone-100 dark:hover:bg-stone-800 transition-colors group"
        >
            <span className="w-7 h-7 flex items-center justify-center rounded-lg bg-stone-100 dark:bg-stone-800 group-hover:bg-topwr group-hover:text-white transition-colors">
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                    className="w-4 h-4"
                >
                    <path d="M10.75 4.75a.75.75 0 00-1.5 0v4.5h-4.5a.75.75 0 000 1.5h4.5v4.5a.75.75 0 001.5 0v-4.5h4.5a.75.75 0 000-1.5h-4.5v-4.5z" />
                </svg>
            </span>
            <span>Nowa rozmowa</span>
        </button>
    );
}
