import csv
import re

def convert_chat_to_csv(input_file, output_file):
    pattern = re.compile(r"^(\d{2}/\d{2}/\d{4}, \d{2}:\d{2}) - (\+\d{10,}): (.+)$")

    messages = []
    current = {"timestamp": "", "mobile_number": "", "message": ""}

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            match = pattern.match(line)
            if match:
                # Save previous message if exists
                if current["timestamp"]:
                    messages.append(current)

                # Start a new message
                timestamp, mobile_number, message = match.groups()
                current = {
                    "timestamp": timestamp,
                    "mobile_number": mobile_number,
                    "message": message
                }
            else:
                # Continuation of the current message
                current["message"] += " " + line

        # Append the last message
        if current["timestamp"]:
            messages.append(current)

    # Write all messages to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['timestamp', 'mobile_number', 'message'])

        for msg in messages:
            writer.writerow([msg["timestamp"], msg["mobile_number"], msg["message"]])

# Example usage
convert_chat_to_csv("GroupChat.txt", "chat_log.csv")
