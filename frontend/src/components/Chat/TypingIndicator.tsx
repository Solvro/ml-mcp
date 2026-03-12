export function TypingIndicator() {
    return (
        <div className="flex w-full mb-4 justify-start">
            <div className="bg-stone-50 dark:bg-stone-800 border border-stone-200 dark:border-stone-700 rounded-2xl rounded-bl-sm px-4 py-3">
                <div className="flex gap-1 items-center h-4">
                    <span className="w-2 h-2 bg-topwr rounded-full animate-bounce [animation-delay:-0.3s]" />
                    <span className="w-2 h-2 bg-topwr rounded-full animate-bounce [animation-delay:-0.15s]" />
                    <span className="w-2 h-2 bg-topwr rounded-full animate-bounce" />
                </div>
            </div>
        </div>
    );
}
