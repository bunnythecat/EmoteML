from tqdm import tqdm
import pysrt
import pickle

subs = pysrt.open("xQc Has a Blast Playing BattleBit Remastered with Jesse & Dizzy.srt")
# subs.shift(seconds=-799)

num_combo = 7
chat_file = "xQc Has a Blast Playing BattleBit Remastered with Jesse & Dizzy.txt"
emote_file = "emotes.txt"
with open(chat_file, 'r', encoding="utf8") as f:
    lines = f.readlines()

with open(emote_file, 'r') as f:
    emotes = f.read().split(',')


def time_2_seconds(t, shifting=0):
    hms = t.split(':')
    return int(hms[0]) * 3600 + int(hms[1]) * 60 + int(hms[2]) + shifting


def get_sub(subs, s, delay=2, last=3):
    ends = s - delay
    parts = subs.slice(ends_before={'seconds': ends})
    if len(parts) > 3:
        return parts[-last:].text.replace("\n", " ")
    elif len(parts) > 1:
        return parts[1:].text.replace("\n", " ")
    else:
        return subs[:1].text.replace("\n", " ")


print("Total emotes: " + str(len(emotes)))
previous_emotes = {}
previous_emote = ""
sentences = []
shifting = -time_2_seconds(lines[0].split('] ')[0].replace('[', ''))
samples = []
new_lines = []
for line in tqdm(lines, total=len(lines), desc='Reading emotes from chat'):
    # loop over all emotes in the list to see if they exist in the chat line
    time = line.split('] ')[0].replace('[', '')
    time = time_2_seconds(time, shifting)
    text = line.split(': ')[-1]
    # new_lines.append([time, text])

    for emote in emotes:
        if emote in text:
            previous_emotes[emote] = previous_emotes.get(emote, 0) + 2
    for key, value in previous_emotes.items():
        value -= 1
        if value <= 0:
            previous_emotes.pop(key)
        elif value >= num_combo:
            if key != previous_emote:
                previous_emote = key
                new_lines.append([time, key])
                sentence = get_sub(subs, time, delay=1, last=3)
                samples.append([sentence, key])
            previous_emotes = {}

print("Number of samples: " + str(len(samples)))

counts = dict()
for sample in samples:
    counts[sample[1]] = counts.get(sample[1], 0) + 1

print(counts)

print(samples[:10])
print(new_lines[:10])
with open("battlebit.csv", "w") as f:
    for sample in samples:
        f.write(",".join(sample) + "\n")

# with open("primitive.p", "wb") as f:
#     pickle.dump(samples, f)
