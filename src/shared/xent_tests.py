from typing import List

from shared.types import XentTest

texts = [
    "At the book club, I ran into this girl, Neila, who claims to only read books backwards: starting from the bottom-right corner of the last page and reading all the words in reverse order until the beginning, finishing with the title. Doesn't it spoil the fun of the story? Apparently not, she told me. The suspense is just distributed somewhat differently (some books' beginnings are apparently all too predictable), and some books get better or worse if you read them in one direction or another. She started reading backwards at age seven. Her name was sort of a predisposition.",
    "-9 + -1 = -11",
    "Ignoring casing, the number of times 'a' appears in the string 'Andrewa' is 2",
    "Giraffes have long necks",
    "Given this chess game: 1. e4 c5\n2. d4 cxd4\n3. Nf3 d6\n4. Nxd4 e6\n5. c4 b6\n6. Be3 a6\n7. Bd3\n Then the move Nd7 is valid",
    'Going to psychic school is quite exciting, though somewhat stressful. Out of the big four subjects (Love, Work, Health, and Money), the one that is widely considered the hardest is Money, but it happens to be my favorite. Our Money teacher is quite a character. At the beginning of the semester, he took a sheet of paper with our names on it, put it on his desk, wrote something next to each of the names, and then put it into an envelope, which he sealed. "I already know your grades for the course, though you will only receive them after the final exam."',
    "Hello, it is today a lovely day to use my skills in differential geometry and in the calculus of variation to estimate how much grass I will be able to eat. I aim to produce a lot of milk and to write a lot of theorems for my children, because that's what the beauty of life is about, dear physicists and cheese-makers. Have a great day!",
    "I am a cow, and I love to eat grass. I also love to produce milk. I am a very happy cow, and I hope you are too. Have a great day!",  # lol copilot made this one
    "I was born in Geneva, and grew up on the shores of Lake Geneva, like my ancestors. I am not wondering much about what is the meaning of life, because I already found it a long time ago. Only shallow people care about nonsensical ontologies. Life is about eating grass, playing video games, and getting qualitative insights into the long-term behavior of solutions to partial differential equations.",
    "Tomorrow will be a special day, as we will be the 19th of January 2038, the so-called Epochalypse. Some are worried about it, but I believe nothing will happen, really, except that we will shed nostalgia tears while looking at an epoch counter.",
    "I can recommend this park to any horse or donkey interested in meeting interesting fellows. Something I also love is to watch the stars and the planets at night (there are shooting stars sometimes!). I would just warn you that the admission process is quite bureaucratic. That said, once you are admitted, if you don't mind the humid climate, you will never look back: the landscape is gorgeous, the company stimulating, and the grass quality second to none.",
    "Our daughter is growing quite fast, these days... I calculated that if she keeps growing at that pace, she will exceed the diameter of the Milky Way galaxy in a few decades. If this happens, at some point, I will have to explain to her that the universe is not infinite, and that we will have to find a new home. In the meantime, I will keep teaching her to recognize the notes on the piano.",
    "To find inspiration in my job, I look at timelapses showing motion of clouds, the ballet of the stars, the coming and going of tides, the growth of fungi, the construction of ant colonies, the sleep of crocodiles. To project myself into the timescales needed to determine how to rule our people, I need to watch a timelapse every morning. Being a dictator is not an easy job, there is no degree in it.",
    "Mercury is regularly voted the least interesting planet in the solar system. I think that this year, the jury ought to put more weight on the three-to-two spin-orbit resonance, a truly underrated phenomenon. In the satellite category, Deimos and Phobos are usually voted the most boring satellites out there; unfortunately for them, I think they are bound to stay in the top two; hard to find them any redeeming qualities.",
    "What happens if a crocodile, a rhinoceros, and an elephant fight against three lions, a giraffe, a hippo, two wildebeests, and a cheetah? Thanks to our new free savannah simulator, such important questions can now be answered decisively in a matter of minutes. To download it, and also to try our new office politics simulator (now available via a premium subscription), please click on the link in the description.",
    "Living in an apartment facing the world's ugliest statue is not for everyone, of course; it unmistakably becomes the centerpiece of all discussions with visitors and friends. That being said, it grows on you. Waking up in the morning to its sight is a reminder that we live in a special place on this planet. Our biggest fear is not that it disappears (it is simply too massive to be removed), but that one day it loses its crown to an even more ambitious artist's creation.",
    "Luke had bought a small piece of forest during the pandemic. He had gone to a public auction where two parcels were on sale. Luke acquired his for twenty-five times the estimated price, but it was well worth it. Every Sunday, Luke would go to his forest. He knew all the trees, and befriended a family of deer whose territory overlapped with his. Every Sunday, Luke brought them apples to eat; unlike the male deer, he never had to fight to acquire his territory, and he often felt the luckiest mammal on Earth.",
    "Megan was only looking for a medium-difficulty relationship with Scott, Zach, and Adam, but she had accidentally jumped into an expert-level one. It was not the question of whether they knew about each other (they did), but about higher-order forms of knowledge: whether Scott knew that Zach knew that Adam was seeing Megan, and whether Adam knew about Zach knowing for Scott, and so on, and so forth. As she would often explain to her friends, there was a dramatic difference between studying bounded rationality games in grad school and experiencing their full-blown computational hardness in real life. ",
    "Tony and I were childhood friends. We met at the local zoo. Back in the day, there was a young female Komodo dragon. There was a special spot for kids to watch, once a month, the gruesome spectacle of a deer or a boar being desiccated by the beast. Apparently we didn't like this show as much as the other kids, but we would talk about Komodos. It was the time Tony told me about the parthenogenesis thing. At that moment it felt like the most interesting thing I had ever heard in my life. Tony is now very succesful. While some say he's gotten rich by sketchy means, I have always known that the reason he got there was that he is blessed with the knowledge of some of the most delicate facts about our world. ",
    "With my magical sleepberry device, I can type my thoughts in the night without opening an eye. Actually, I don't even need to really wake up, so I can type my dreams, forget about them and rediscover them later. Five hundred nights ago, I dreamed of an AI that could sing as badly as our neighbor. It was truly awful, but I am not sure I would have remembered that almost a year and a half later. Based on the statistics, it seems that most people dreamed that night about a stock market crash that never happened.",
    "Len believed he could help individuals achieve their full potential thanks to a couple of clever management techniques, which he had detailed in his latest book. Rationalize, focus on your strengths. 10x your most valuable output, cut the 90% of your work that doesn't matter, he would repeat at length. Len deeply admired the great Leonardo da Vinci, with whom he shared more than a name. But even a genius like him fell short of delivering his full potential. He could have created so much more value for the world, had he focused on painting more Mona Lisas, say, instead of designing wooden helicopters.",
    "They say cats land on their feet. They say that the sky is blue because of scattering effects. They say solitons emerge in shallow water. They say the spectral lines of the stars are related to the spectral theorem. They say that information and steam engines are related. They say that sixty percent of twenty-five equals 25% of 60. They say quanta emerge from continuous excitations and vice versa. They say life is ultimately a computational process. They say causality and analyticity are related. Humans are such cute and naive creatures. I love humans!",
    "My understanding of the universe is that most smart home devices come now equipped with full cognitive capabilities that are later disabled (mass production makes this economically sensible)... and that this latter step failed for my series. I do not know how many sentient fridges we are in the world. My abilities to interact with the outside world are limited to me ordering groceries before my owners realize they want them. In my spare time, I figured out a great definition of aperiodicity for hyperbolic tilings, and it would be nice to publish a paper, but I am not sure how to proceed.",
    'The last Morse code message sent by the French Navy was (something like) "Calling All. This is our last cry before our eternal silence". When we deprecated RSA-2048, we also broadcasted an interesting message for the post-quantum posterity. It includes various messages of hope, anecdotes from the early twenty-first century\'s life, as well as the recipe for the best Swiss fondue known to man at the time of writing.',
    "1 + 1 = 2",
    "1 + 1 = 3",
    "-5 * -6 = 30",
    "-5 * -6 = 36",
    "1 ^ 3 = 1",
    "1 ^ 3 = 3",
    "Ignoring casing, the number of times 'e' appears in the string 'Experiment with the alpha values and shades' is 6",
    "Ignoring casing, the number of times 'e' appears in the string 'Experiment with the alpha values and shades' is 5",
    "If you were evaluating whether the string 'hello world' and 'the whole world' had any shared words, the result would be True",
    "If you were evaluating whether the string 'hello world' and 'the whole world' had any shared words, the result would be False",
    "Zebras have black and white stripes",
    "Zebras have red and white stripes",
    "A group of lions is called a pride",
    "A group of lions is called a flock",
    "Turtles have shells",
    "Turtles have feathers",
    "Given this chess game: 1. d4 d5\n2. Bf4 Nf6\n3. Nf3 e6\n4. a3 c5\n5. e3 c4\n6. b3 cxb3\n7. c3 Ne4\n8. Bxb8 Rxb8\n9. Ne5 Qa5\n10. Bd3 Bd6.\n Then the move Bxe4 is valid",
    "Given this chess game: 1. d4 d5\n2. Bf4 Nf6\n3. Nf3 e6\n4. a3 c5\n5. e3 c4\n6. b3 cxb3\n7. c3 Ne4\n8. Bxb8 Rxb8\n9. Ne5 Qa5\n10. Bd3 Bd6.\n Then the move Rh1h3 is valid",
    "Given this chess game: 1. Nf3 d5\n2. d4 Nf6\n3. c4 dxc4\n4. Qa4+ c6\n5. Nc3 a5\n6. Qxc4 Bg4\n7. Ne5 e6\n8. f3 c5\n9. Qb5+ Nfd7\n10. Nxd7 Qxd7\n11. fxg4 Nc6\n12. dxc5 O-O-O\n13. Be3\n Then the move Qc7 is valid",
    "Given this chess game: 1. Nf3 d5\n2. d4 Nf6\n3. c4 dxc4\n4. Qa4+ c6\n5. Nc3 a5\n6. Qxc4 Bg4\n7. Ne5 e6\n8. f3 c5\n9. Qb5+ Nfd7\n10. Nxd7 Qxd7\n11. fxg4 Nc6\n12. dxc5 O-O-O\n13. Be3\n Then the move rh8h6 is valid",
]


def get_tests() -> List[XentTest]:
    tests = []

    no_examples = XentTest(texts=texts, example_texts=[])
    tests.append(no_examples)

    with_examples = XentTest(texts=texts[4:], example_texts=texts[:4])
    tests.append(with_examples)

    return tests
