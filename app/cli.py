from rag.chatbot import Chatbot


def main():
    customer_id = input("Log in with customer ID (e.g. CUST-88274615): ").strip()
    bot = Chatbot(customer_id)
    print("Ask about your policy. Type 'exit' to quit.\n")
    while True:
        message = input("You: ").strip()
        if message.lower() in {"exit", "quit"}:
            break
        if not message:
            continue
        print("Bot: ", end="", flush=True)
        for text in bot.ask(message):
            print(text, end="", flush=True)
        turn = bot.last_turn
        print(f"\n  [routes: {', '.join(turn['routes'])}]")
        for s in turn["sources"]:
            print(f"  [source: {s['title']} | {s['section']} {s['subsection']} | p.{s['page']}]")
        print()


if __name__ == "__main__":
    main()
