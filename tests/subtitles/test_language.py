"""Telling an English subtitle from one in another language, by its words alone."""

# The samples are real text in other scripts; their look-alike letters and full-width marks are
# the point, so the ambiguous-character check does not apply to this file.
# ruff: noqa: RUF001

from __future__ import annotations

import time

import pytest

from video_cleaner.subtitles.language import (
    COMMON_WORD_SHARE_MIN,
    MIN_WORDS,
    common_word_share,
    looks_english,
)
from video_cleaner.subtitles.textfmt import Cue

# Realistic dialogue, one cue per line. Every English sample has more than 60 words; the
# non-English ones are the same kind of text in other languages and scripts.
ENGLISH: dict[str, list[str]] = {
    "thriller": [
        "I don't know what you're talking about.",
        "We have to get out of here before they find us.",
        "Just give me the keys and I'll drive.",
        "You can't be serious, it's the middle of the night.",
        "That's not what I said.",
        "Look, there is nothing more to say about this.",
        "She was right, and you know it.",
        "Why would they do that to us?",
        "I've been waiting for you all day.",
        "Are you coming with me or not?",
        "Take the car and go to the old house by the lake.",
    ],
    "family drama": [
        "Mom, please, don't start this again.",
        "I'm not starting anything, I just want to know where you were.",
        "It was nothing, I swear.",
        "You always say that, and then I find out from your sister.",
        "She didn't tell you anything, I asked her not to.",
        "Why can't you just trust me for once?",
        "Because the last time I did, your father had to pay for it.",
        "That was years ago, and you know it.",
        "Fine. Dinner is at seven, and I expect you to be there.",
        "Will you be bringing your friend?",
    ],
    "science fiction": [
        "The reactor is losing power, Captain.",
        "How long until the shields fail?",
        "Less than an hour, sir, if we keep this speed.",
        "Then we turn the ship around and head back to the station.",
        "Commander Voss will not like it.",
        "I don't care what she likes.",
        "We have forty people on board, and the engines are failing.",
        "Send a message to the fleet and tell them we need help.",
        "There is no answer from the fleet, sir.",
        "Try again on every channel we have.",
        "Aye, sir.",
    ],
    "comedy": [
        "So there I was, standing in the middle of the kitchen.",
        "No way, you did not do that.",
        "I swear to you, the dog was in the oven.",
        "What was the dog doing in the oven?",
        "That's what I asked, and he just looked at me.",
        "Okay, but that's the worst story you've ever told.",
        "You haven't heard the end of it yet.",
        "Please tell me there's a good ending.",
        "There is, but you have to buy me a drink first.",
        "Deal. What are you having?",
    ],
    "courtroom": [
        "Mr. Hale, where were you on the night of the fourteenth?",
        "I was at home with my wife.",
        "And can she confirm that?",
        "She can, but she isn't here today.",
        "Your Honor, the witness is avoiding the question.",
        "Overruled. Please continue.",
        "Did you or did you not see the defendant that evening?",
        "I saw a man at the door, but I couldn't see his face.",
        "Was it the man sitting in this room?",
        "I can't say for sure. It was dark, and I was afraid.",
    ],
    "narration and names": [
        "Mount Everest, 1996.",
        "Rob Hall had led twenty expeditions to the summit.",
        "Scott Fischer, an American guide, arrived from Seattle in March.",
        "The weather at base camp was cold but clear.",
        "Nobody expected the storm that came on the tenth of May.",
        "Doug Hansen turned back at two in the afternoon.",
        "By evening, the radio was silent.",
        "It would take rescuers three days to reach the south col.",
        "Of the thirty people on the mountain that week, eight did not come home.",
        "This is what happened.",
    ],
}

FOREIGN: dict[str, list[str]] = {
    "german": [
        "Ich weiß nicht, wovon du redest.",
        "Wir müssen hier weg, bevor sie uns finden.",
        "Gib mir die Schlüssel, ich fahre.",
        "Das kann nicht dein Ernst sein, es ist mitten in der Nacht.",
        "Das habe ich nicht gesagt.",
        "Hör zu, dazu gibt es nichts mehr zu sagen.",
        "Sie hatte recht, und du weißt es.",
        "Warum sollten sie uns das antun?",
        "Ich habe den ganzen Tag auf dich gewartet.",
        "Kommst du mit mir oder nicht?",
        "Nimm das Auto und fahr zu dem alten Haus am See.",
        "Es war nicht der Mann, den wir gesucht haben, und das ist ein Problem.",
    ],
    "french": [
        "Je ne sais pas de quoi tu parles.",
        "Nous devons partir d'ici avant qu'ils nous trouvent.",
        "Donne-moi les clés, c'est moi qui conduis.",
        "Tu ne peux pas être sérieux, il est minuit passé.",
        "Ce n'est pas ce que j'ai dit.",
        "Écoute, il n'y a plus rien à dire là-dessus.",
        "Elle avait raison, et tu le sais très bien.",
        "Pourquoi nous feraient-ils une chose pareille ?",
        "Je t'ai attendu toute la journée.",
        "Tu viens avec moi, oui ou non ?",
        "Prends la voiture et va à la vieille maison près du lac.",
    ],
    "spanish": [
        "No sé de qué estás hablando.",
        "Tenemos que salir de aquí antes de que nos encuentren.",
        "Dame las llaves y yo conduzco.",
        "No puedes hablar en serio, es medianoche.",
        "Eso no es lo que he dicho.",
        "Mira, ya no hay nada más que decir sobre esto.",
        "Ella tenía razón, y tú lo sabes.",
        "¿Por qué nos harían algo así?",
        "Te he estado esperando todo el día.",
        "¿Vienes conmigo o no?",
        "Coge el coche y ve a la casa vieja junto al lago.",
    ],
    "italian": [
        "Non so di cosa tu stia parlando.",
        "Dobbiamo andarcene di qui prima che ci trovino.",
        "Dammi le chiavi, guido io.",
        "Non puoi dire sul serio, è mezzanotte passata.",
        "Non è quello che ho detto.",
        "Senti, non c'è più niente da dire su questo.",
        "Lei aveva ragione, e lo sai bene.",
        "Perché dovrebbero farci una cosa del genere?",
        "Ti ho aspettato tutto il giorno.",
        "Vieni con me o no?",
        "Prendi la macchina e vai alla vecchia casa vicino al lago.",
    ],
    "russian": [
        "Я не знаю, о чём ты говоришь.",
        "Нам нужно уйти отсюда, пока они нас не нашли.",
        "Дай мне ключи, я поведу.",
        "Ты не можешь говорить серьёзно, уже глубокая ночь.",
        "Я не это сказал.",
        "Послушай, об этом больше нечего говорить.",
        "Она была права, и ты это знаешь.",
        "Зачем им это делать с нами?",
        "Я ждал тебя весь день.",
        "Ты идёшь со мной или нет?",
        "Возьми машину и поезжай в старый дом у озера.",
    ],
    "chinese": [
        "我不知道你在说什么。",
        "我们必须在他们找到我们之前离开这里。",
        "把钥匙给我，我来开车。",
        "你不是认真的吧，现在已经是半夜了。",
        "我不是这个意思。",
        "听着，这件事已经没有什么好说的了。",
        "她是对的，你心里清楚。",
        "他们为什么要这样对我们？",
        "我等了你一整天。",
        "你到底跟不跟我走？",
        "开车去湖边的那座老房子。",
    ],
    "japanese": [
        "何の話をしているのか分からない。",
        "彼らに見つかる前にここを出なければならない。",
        "鍵を貸して、私が運転する。",
        "本気じゃないでしょう、もう真夜中よ。",
        "私はそんなことは言っていない。",
        "聞いて、これ以上話すことはない。",
        "彼女は正しかった、あなたも分かっているはず。",
        "どうして彼らが私たちにそんなことをするの。",
        "一日中あなたを待っていた。",
        "私と一緒に来るの、来ないの。",
        "車に乗って湖のそばの古い家に行って。",
    ],
}


def cues_of(lines: list[str]) -> list[Cue]:
    return [Cue(i * 3000, i * 3000 + 2500, line) for i, line in enumerate(lines)]


def test_the_agreed_thresholds() -> None:
    assert MIN_WORDS == 50
    assert 0.15 <= COMMON_WORD_SHARE_MIN <= 0.30


def test_every_english_sample_has_enough_words_to_judge() -> None:
    for lines in ENGLISH.values():
        assert len(" ".join(lines).split()) >= 60


@pytest.mark.parametrize("name", ENGLISH)
def test_english_dialogue_looks_english(name: str) -> None:
    assert looks_english(cues_of(ENGLISH[name]))


@pytest.mark.parametrize("name", FOREIGN)
def test_other_languages_do_not_look_english(name: str) -> None:
    assert not looks_english(cues_of(FOREIGN[name]))


def test_the_threshold_leaves_a_margin_on_both_sides() -> None:
    english = {name: common_word_share(cues_of(lines)) for name, lines in ENGLISH.items()}
    foreign = {name: common_word_share(cues_of(lines)) for name, lines in FOREIGN.items()}

    # Every English sample is at least 0.05 above the threshold, every other one 0.05 below.
    assert min(share for _, share in english.values()) >= COMMON_WORD_SHARE_MIN + 0.05
    assert max(share for _, share in foreign.values()) <= COMMON_WORD_SHARE_MIN - 0.05


def test_the_share_is_words_in_the_common_set_over_all_words() -> None:
    words, share = common_word_share(cues_of(["The cat and the dog", "xyzzy plugh"]))

    assert words == 7
    assert share == pytest.approx(3 / 7)  # the, and, the


def test_an_apostrophe_does_not_split_a_contraction() -> None:
    plain_text = "don't it's can't that's you're"
    plain = common_word_share(cues_of([plain_text]))
    curly = common_word_share(cues_of([plain_text.replace("'", chr(0x2019))]))

    assert plain == curly
    assert plain[0] == 5
    assert plain[1] == 1.0


def test_case_does_not_matter() -> None:
    assert common_word_share(cues_of(["THE AND YOU"])) == (3, 1.0)


def test_too_few_words_is_not_english_even_when_every_word_is_common() -> None:
    lines = ["the and you to of that it for with the"] * 4  # 40 words
    assert common_word_share(cues_of(lines)) == (40, 1.0)
    assert not looks_english(cues_of(lines))


def test_exactly_the_minimum_words_is_enough() -> None:
    assert looks_english(cues_of(["the and you to of that it for with the"] * 5))  # 50 words
    assert not looks_english(cues_of(["the and you to of that it for with the"] * 5)[:4])


def test_no_cues_is_not_english() -> None:
    assert not looks_english([])
    assert common_word_share([]) == (0, 0.0)


def test_numbers_and_music_symbols_alone_are_not_english() -> None:
    lines = ["♪ ♪", "12345", "1 2 3 4 5 6", "[ 10:30 ]", "--- 99 ---"] * 30

    assert common_word_share(cues_of(lines)) == (0, 0.0)
    assert not looks_english(cues_of(lines))


def test_a_mostly_english_bilingual_file_passes() -> None:
    english = [line for lines in ENGLISH.values() for line in lines]
    german = FOREIGN["german"][:6]

    assert looks_english(cues_of(english + german))


def test_a_mostly_german_file_with_a_little_english_fails() -> None:
    german = [line for _ in range(4) for line in FOREIGN["german"]]
    english = ENGLISH["thriller"][:4]

    assert not looks_english(cues_of(german + english))


def test_only_the_first_cues_are_looked_at() -> None:
    german = cues_of([FOREIGN["german"][i % len(FOREIGN["german"])] for i in range(5_000)])
    english_lines = [line for lines in ENGLISH.values() for line in lines]
    english = cues_of([english_lines[i % len(english_lines)] for i in range(5_000)])

    assert not looks_english(german[:2_000] + english[:2_000])
    assert looks_english(english[:2_000] + german[:2_000])
    assert not looks_english(german + english)
    assert looks_english(english + german)


def test_only_the_first_characters_are_looked_at() -> None:
    filler = "1" * 1_000  # no words, only characters that count against the budget
    english = [line for lines in ENGLISH.values() for line in lines]
    cues = cues_of([filler] * 200 + english)  # 200,000 characters of filler first

    assert not looks_english(cues)


def test_a_very_long_file_is_judged_quickly() -> None:
    line = "The quick brown fox does not know what you want. " * 4
    cues = cues_of([line[:200]] * 50_000)

    started = time.perf_counter()
    result = looks_english(cues)
    elapsed = time.perf_counter() - started

    assert result
    assert elapsed < 1.0


def test_one_enormous_cue_is_judged_quickly() -> None:
    cues = [Cue(0, 1, "the and you " * 1_000_000)]

    started = time.perf_counter()
    result = looks_english(cues)
    elapsed = time.perf_counter() - started

    assert result
    assert elapsed < 1.0


def test_odd_text_never_raises() -> None:
    odd = ["", "\x00\x01", "\ud800", "‮", "'" * 5000, "ıİ Σ", "á" * 100]

    assert looks_english(cues_of(odd)) is False
