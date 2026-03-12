import { type KeyboardEvent, useRef, useState } from "react";

interface InputBarProps {
    onSend: (text: string) => void;
    disabled?: boolean;
}

export function InputBar({ onSend, disabled = false }: InputBarProps) {
    const [value, setValue] = useState("");
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit();
        }
    }

    function submit() {
        const text = value.trim();
        if (!text || disabled) return;
        onSend(text);
        setValue("");
        if (textareaRef.current) {
            textareaRef.current.style.height = "auto";
        }
    }

    function handleInput() {
        const el = textareaRef.current;
        if (!el) return;
        el.style.height = "auto";
        el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
    }

    return (
        <div className="border-t border-stone-100 dark:border-stone-800 p-4 bg-white dark:bg-stone-900">
            <div className="flex items-end gap-3 bg-white dark:bg-stone-800 border border-stone-200 dark:border-stone-700 rounded-2xl px-4 py-3 focus-within:border-topwr transition-colors">
                <textarea
                    ref={textareaRef}
                    value={value}
                    onChange={(e) => setValue(e.target.value)}
                    onKeyDown={handleKeyDown}
                    onInput={handleInput}
                    disabled={disabled}
                    placeholder="Zadaj pytanie o PWr..."
                    rows={1}
                    className="flex-1 bg-transparent text-stone-800 dark:text-stone-100 placeholder-stone-400 dark:placeholder-stone-500 resize-none outline-none text-sm leading-relaxed max-h-40 disabled:opacity-50"
                />
                <button
                    onClick={submit}
                    disabled={disabled || !value.trim()}
                    className="shrink-0 w-8 h-8 flex items-center justify-center rounded-lg bg-topwr text-white hover:bg-topwr-hover disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                    aria-label="Wyślij wiadomość"
                >
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 20 20"
                        fill="currentColor"
                        className="w-4 h-4"
                    >
                        <path d="M3.105 2.289a.75.75 0 00-.826.95l1.414 4.925A1.5 1.5 0 005.135 9.25h6.115a.75.75 0 010 1.5H5.135a1.5 1.5 0 00-1.442 1.086l-1.414 4.926a.75.75 0 00.826.95 28.896 28.896 0 0015.293-7.154.75.75 0 000-1.115A28.897 28.897 0 003.105 2.289z" />
                    </svg>
                </button>
            </div>
            <p className="text-xs text-stone-400 dark:text-stone-500 text-center mt-2">
                Enter — wyślij · Shift+Enter — nowa linia
            </p>
        </div>
    );
}
