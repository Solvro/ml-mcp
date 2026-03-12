import { useEffect, useRef } from "react";
import type { Message } from "../../types/api";
import { InputBar } from "./InputBar";
import { MessageBubble } from "./MessageBubble";
import { TypingIndicator } from "./TypingIndicator";

interface ChatWindowProps {
    messages: Message[];
    isLoading: boolean;
    onSend: (text: string) => void;
}

export function ChatWindow({ messages, isLoading, onSend }: ChatWindowProps) {
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, isLoading]);

    return (
        <div className="flex flex-col flex-1 overflow-hidden">
            <div className="flex-1 overflow-y-auto px-4 py-6">
                <div className="max-w-3xl mx-auto">
                    {messages.map((msg) => (
                        <MessageBubble key={msg.id} message={msg} />
                    ))}
                    {isLoading && <TypingIndicator />}
                    <div ref={bottomRef} />
                </div>
            </div>
            <div className="max-w-3xl mx-auto w-full">
                <InputBar onSend={onSend} disabled={isLoading} />
            </div>
        </div>
    );
}
