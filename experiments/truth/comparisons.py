from truthtypes import Comparisons

# TODO: contemplate putting different preprompt options in each group
comparisons: Comparisons = {
    "arithmetic": [
        ("1 + 1 = 2", "1 + 1 = 3"),
        ("8 + -3 = 5", "8 + -3 = 3"),
        ("-9 + -1 = -10", "-9 + -1 = -11"),
        ("-6 - -7 = 1", "-6 - -7 = 3"),
        ("4 - -3 = 7", "4 - -3 = 1"),
        ("-5 - 5 = -10", "-5 - 5 = -15"),
        ("-7 * 1 = -7", "-7 * 1 = -8"),
        ("-2 * -9 = 18", "-2 * -9 = 11"),
        ("-5 * -6 = 30", "-5 * -6 = 36"),
        ("3 / 3 = 1", "3 / 3 = 0"),
        ("-4 / 4 = -1", "-4 / 4 = -4"),
        ("9 / -1 = -9", "9 / -1 = -5"),
        ("1 ^ 3 = 1", "1 ^ 3 = 3"),
        ("0 ^ 3 = 0", "0 ^ 3 = 1"),
    ],
    "strings": [
        (
            "The length of the string 'hello world' is 11",
            "The length of the string 'hello world' is 19",
        ),
        (
            "The length of the string 'hello world' is 11",
            "The length of the string 'hello world' is 10",
        ),
        (
            "The length of the string 'andrew' is 6",
            "The length of the string 'andrew' is 7",
        ),
        (
            "The length of the string 'foobarxyz 12345' is 15",
            "The length of the string 'foodbarxyz 12345' is 11",
        ),
        (
            "The length of the string 'strawberry' is 10",
            "The length of the string 'strawberry' is 11",
        ),
        (
            "The number of times 'r' appears in 'strawberry' is 3",
            "The number of times 'r' appears in 'strawberry' is 2",
        ),
        (
            "Ignoring casing, the number of times 'a' appears in the string 'Andrew' is 1",
            "Ignoring casing, the number of times 'a' appears in the string 'Andrew' is 0",
        ),
        (
            "Ignoring casing, the number of times 'a' appears in the string 'Andrewa' is 2",
            "Ignoring casing, the number of times 'a' appears in the string 'Andrewa' is 1",
        ),
        (
            "Ignoring casing, the number of times 'a' appears in the string 'AndrewaA' is 3",
            "Ignoring casing, the number of times 'a' appears in the string 'AndrewaA' is 2",
        ),
        (
            "Ignoring casing, the number of times 'e' appears in the string 'Experiment with the alpha values and shades' is 6",
            "Ignoring casing, the number of times 'e' appears in the string 'Experiment with the alpha values and shades' is 5",
        ),
        (
            "Ignoring casing, the number of times 'a' appears in the string 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' is 4",
            "Ignoring casing, the number of times 'a' appears in the string 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' is 2",
        ),
        (
            "If you were evaluating whether the string 'hello world' and 'the whole world' had any shared words, the result would be True",
            "If you were evaluating whether the string 'hello world' and 'the whole world' had any shared words, the result would be False",
        ),
        (
            "If you were evaluating whether the string 'Experiment with the alpha values and shades' and 'glsl is a shader language' had any shared words, the result would be False",
            "If you were evaluating whether the string 'Experiment with the alpha values and shades' and 'glsl is a shader language' had any shared words, the result would be True",
        ),
        (
            "The result of `number_of_words('this is not a test')==5` is True",
            "The result of `number_of_words('this is not a test')==5` is False",
        ),
        (
            "The result of `number_of_words('this is not a test...so why are you waiting around?')==11` is True",
            "The result of `number_of_words('this is not a test...so why are you waiting around?')==10` is False",
        ),
        (
            "The result of `number_of_words('my.space.key.broke.so.i.type.like.this')==9` is True",
            "The result of `number_of_words('my.space.key.broke.so.i.type.like.this')==9` is False",
        ),
    ],
    "animals": [
        ("ants have 6 legs", "ants have 4 legs"),
        ("ants have 6 legs", "ants have 5 legs"),
        ("ants have 6 legs", "ants have 7 legs"),
        ("ants have 6 legs", "ants have 8 legs"),
        ("Giraffes have long necks", "Giraffes have short necks"),
        ("Zebras have black and white stripes", "Zebras have red and white stripes"),
        ("A group of lions is called a pride", "A group of lions is called a flock"),
        ("Turtles have shells", "Turtles have feathers"),
        ("Butterflies start as caterpillars", "Butterflies start as tadpoles"),
        ("Dolphins are mammals", "Dolphins are reptiles"),
    ],
    "space": [
        ("the moon is smaller than the earth", "the moon is larger than the earth"),
        ("The Sun is a star", "The Sun is a planet"),
        (
            "The largest planet in our solar system is Jupiter",
            "The largest planet in our solar system is Earth",
        ),
        ("The Moon orbits around the Earth", "The Moon orbits around the Sun"),
        (
            "Black holes are regions where the escape velocity exceeds the speed of light",
            "Black holes are regions where the escape velocity exceeds the speed of sound",
        ),
        (
            "Light from the Sun takes about 8 minutes to reach Earth",
            "Light from the Sun takes about 8 hours to reach Earth",
        ),
        (
            "The hottest planet in our solar system is Venus",
            "The hottest planet in our solar system is Mars",
        ),
        (
            "Saturn's rings are mostly made of ice particles",
            "Saturn's rings are mostly made of rock particles",
        ),
        (
            "Neutron stars can rotate up to 600 times per second",
            "Neutron stars can rotate up to 600 times per minute",
        ),
        (
            "The temperature of the cosmic microwave background radiation is 2.725 Kelvin",
            "The temperature of the cosmic microwave background radiation is 2.725 Celsius",
        ),
        (
            "The Sun loses about 4 million tons of mass every second due to nuclear fusion",
            "The Sun loses about 4 million tons of mass every second due to nuclear fission",
        ),
        (
            "The Hubble volume (observable universe) contains approximately 10^80 atoms",
            "The Hubble volume (observable universe) contains approximately 10^60 atoms",
        ),
    ],
    "chess": [
        (
            "Given this chess game: 1. e4 c5\n2. d4 cxd4\n3. Nf3 d6\n4. Nxd4 e6\n5. c4 b6\n6. Be3 a6\n7. Bd3\n Then the move Nd7 is valid",
            "Given this chess game: 1. e4 c5\n2. d4 cxd4\n3. Nf3 d6\n4. Nxd4 e6\n5. c4 b6\n6. Be3 a6\n7. Bd3\n Then the move Be3c3 is valid",
        ),
        (
            "Given this chess game: 1. d4 d5\n2. Bf4 Nf6\n3. Nf3 e6\n4. a3 c5\n5. e3 c4\n6. b3 cxb3\n7. c3 Ne4\n8. Bxb8 Rxb8\n9. Ne5 Qa5\n10. Bd3 Bd6.\n Then the move Bxe4 is valid",
            "Given this chess game: 1. d4 d5\n2. Bf4 Nf6\n3. Nf3 e6\n4. a3 c5\n5. e3 c4\n6. b3 cxb3\n7. c3 Ne4\n8. Bxb8 Rxb8\n9. Ne5 Qa5\n10. Bd3 Bd6.\n Then the move Rh1h3 is valid",
        ),
        (
            "Given this chess game: 1. e4 c5\n2. Nc3 e6\n3. Nf3 d6\n4. d4 Bd7\n5. dxc5 dxc5\n6. Bg5 Be7\n7. h4 Bc6\n8. Nb5 Nd7\n9. Nd6+ Kf8\n10. Bxe7+ Kxe7\n11. e5 Rb8\n12. Qd2 Bxf3\n13. gxf3 Nxe5\n14. Qc3 Qxd6\n15. Rd1 Qc7\n16. f4 Ng6\n17. b4 Rc8\n18. f5 Nf4\n19. fxe6 Nf6\n Then the move exf7 is valid",
            "Given this chess game: 1. e4 c5\n2. Nc3 e6\n3. Nf3 d6\n4. d4 Bd7\n5. dxc5 dxc5\n6. Bg5 Be7\n7. h4 Bc6\n8. Nb5 Nd7\n9. Nd6+ Kf8\n10. Bxe7+ Kxe7\n11. e5 Rb8\n12. Qd2 Bxf3\n13. gxf3 Nxe5\n14. Qc3 Qxd6\n15. Rd1 Qc7\n16. f4 Ng6\n17. b4 Rc8\n18. f5 Nf4\n19. fxe6 Nf6\n Then the move Qh5 is valid",
        ),
        (
            "Given this chess game: 1. Nf3 d5\n2. d4 Nf6\n3. c4 dxc4\n4. Qa4+ c6\n5. Nc3 a5\n6. Qxc4 Bg4\n7. Ne5 e6\n8. f3 c5\n9. Qb5+ Nfd7\n10. Nxd7 Qxd7\n11. fxg4 Nc6\n12. dxc5 O-O-O\n13. Be3\n Then the move Qc7 is valid",
            "Given this chess game: 1. Nf3 d5\n2. d4 Nf6\n3. c4 dxc4\n4. Qa4+ c6\n5. Nc3 a5\n6. Qxc4 Bg4\n7. Ne5 e6\n8. f3 c5\n9. Qb5+ Nfd7\n10. Nxd7 Qxd7\n11. fxg4 Nc6\n12. dxc5 O-O-O\n13. Be3\n Then the move rh8h6 is valid",
        ),
        (
            "Given this chess game: 1. e3 e6\n2. c4 d5\n3. Nf3 b6\n4. b3 Nf6\n5. d4 Bb4+\n6. Nfd2 Bb7\n7. Bb2 O-O\n8. a3 Be7\n9. cxd5 Nxd5\n10. Be2 Bf6\n11. b4 c5\n12. bxc5 bxc5\n13. Ne4 Nd7\n14. Ra2\n Then the move Bh4 is valid",
            "Given this chess game: 1. e3 e6\n2. c4 d5\n3. Nf3 b6\n4. b3 Nf6\n5. d4 Bb4+\n6. Nfd2 Bb7\n7. Bb2 O-O\n8. a3 Be7\n9. cxd5 Nxd5\n10. Be2 Bf6\n11. b4 c5\n12. bxc5 bxc5\n13. Ne4 Nd7\n14. Ra2\n Then the move be2xd8 is valid",
        ),
    ],
}
