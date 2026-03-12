import type { Message } from "../../types/api";

interface MessageBubbleProps {
    message: Message;
}

const dateFormatter = new Intl.DateTimeFormat("pl-PL", {
    dateStyle: "short",
    timeStyle: "short",
});

export function MessageBubble({ message }: MessageBubbleProps) {
    const isUser = message.role === "user";
    const time = dateFormatter.format(new Date(message.timestamp));

    return (
        <div className={`flex w-full mb-4 ${isUser ? "justify-end" : "justify-start"}`}>
            <div
                className={`flex flex-col gap-1 ${isUser ? "items-end" : "items-start"} max-w-[80%]`}
            >
                <div
                    className={`px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap break-words ${
                        isUser
                            ? "bg-topwr text-white rounded-br-sm"
                            : "bg-stone-50 dark:bg-stone-800 text-stone-800 dark:text-stone-100 border border-stone-200 dark:border-stone-700 rounded-bl-sm"
                    }`}
                >
                    {message.content}
                </div>
                <span className="text-xs text-stone-400 dark:text-stone-500 px-1">{time}</span>
            </div>
        </div>
    );
}
