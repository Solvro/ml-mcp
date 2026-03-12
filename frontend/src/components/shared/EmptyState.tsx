interface EmptyStateProps {
    onSend: (text: string) => void;
}

const SUGGESTIONS = [
    "Kto wykłada analizę matematyczną?",
    "Jakie kierunki oferuje Wydział Informatyki?",
    "Co to jest ToPWR?",
    "Jakie artykuły opublikował Wydział Matematyki?",
];

export function EmptyState({ onSend }: EmptyStateProps) {
    return (
        <div className="flex flex-col flex-1 overflow-hidden bg-white dark:bg-stone-900">
            <div className="flex flex-col flex-1 items-center justify-center px-6 gap-8">
                <div className="text-center">
                    <h1 className="text-3xl font-bold tracking-tight text-stone-900 dark:text-stone-100 mb-2">
                        PWr<span className="text-topwr">Chat</span>
                    </h1>
                    <p className="text-stone-500 dark:text-stone-400 text-sm max-w-sm">
                        Zadaj pytanie o Politechnikę Wrocławską — kursy, wykładowców, wydziały i
                        artykuły naukowe.
                    </p>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-xl">
                    {SUGGESTIONS.map((text) => (
                        <button
                            key={text}
                            onClick={() => onSend(text)}
                            className="text-left text-sm text-stone-600 dark:text-stone-300 bg-stone-50 dark:bg-stone-800 hover:bg-stone-100 dark:hover:bg-stone-700 border border-stone-200 dark:border-stone-700 rounded-xl px-4 py-3 transition-colors"
                        >
                            {text}
                        </button>
                    ))}
                </div>
            </div>
            <div className="max-w-3xl mx-auto w-full px-0">
                <div className="border-t border-stone-100 dark:border-stone-800 p-4 bg-white dark:bg-stone-900">
                    <div className="flex items-end gap-3 bg-white dark:bg-stone-800 border border-stone-200 dark:border-stone-700 rounded-2xl px-4 py-3 focus-within:border-topwr transition-colors">
                        <textarea
                            placeholder="Zadaj pytanie o PWr..."
                            rows={1}
                            onKeyDown={(e) => {
                                if (e.key === "Enter" && !e.shiftKey) {
                                    e.preventDefault();
                                    const text = e.currentTarget.value.trim();
                                    if (text) {
                                        onSend(text);
                                        e.currentTarget.value = "";
                                    }
                                }
                            }}
                            className="flex-1 bg-transparent text-stone-800 dark:text-stone-100 placeholder-stone-400 dark:placeholder-stone-500 resize-none outline-none text-sm leading-relaxed"
                        />
                    </div>
                    <p className="text-xs text-stone-400 dark:text-stone-500 text-center mt-2">
                        Enter — wyślij · Shift+Enter — nowa linia
                    </p>
                </div>
            </div>
        </div>
    );
}
