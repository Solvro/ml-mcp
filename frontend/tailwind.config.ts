import type { Config } from "tailwindcss";

export default {
    content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
    darkMode: "class",
    theme: {
        extend: {
            colors: {
                topwr: {
                    DEFAULT: "#F97316",
                    hover: "#EA6C0A",
                    muted: "#FFF7ED",
                },
            },
        },
    },
    plugins: [],
} satisfies Config;
