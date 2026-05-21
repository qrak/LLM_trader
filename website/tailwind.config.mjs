/** @type {import('tailwindcss').Config} */
export default {
    content: ["./src/**/*.{astro,html,js,jsx,ts,tsx,md,mdx}"],
    theme: {
        extend: {
            colors: {
                shell: {
                    950: "#06080c",
                    900: "#0a0f17",
                    800: "#101826"
                },
                signal: {
                    cyan: "#47d5ff",
                    mint: "#52f2b8",
                    amber: "#ffd166"
                }
            },
            boxShadow: {
                panel: "0 18px 60px -30px rgba(71, 213, 255, 0.35)",
                ring: "0 0 0 1px rgba(148, 163, 184, 0.25), 0 0 0 8px rgba(71, 213, 255, 0.08)"
            },
            borderRadius: {
                xl2: "1.25rem"
            },
            keyframes: {
                pulseTelemetry: {
                    "0%, 100%": { opacity: "0.45", transform: "scale(1)" },
                    "50%": { opacity: "1", transform: "scale(1.08)" }
                },
                streamShift: {
                    "0%": { transform: "translateX(-12px)" },
                    "100%": { transform: "translateX(12px)" }
                }
            },
            animation: {
                telemetry: "pulseTelemetry 1.8s ease-in-out infinite",
                stream: "streamShift 2.6s ease-in-out infinite alternate"
            }
        }
    },
    plugins: []
};