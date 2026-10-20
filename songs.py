"""Songs used to test the instrument. All use only our 7 chords (key of C major).

Each song is a list of chord names, one per bar. The gesture to show for each
chord is the chord's number in audio.CHORDS (C=1, Dm=2, Em=3, F=4, G=5, Am=6, Bdim=7).
"""
from audio import CHORDS

SONGS = {
    "Let It Be (verse)": ["C", "G", "Am", "F", "C", "G", "F", "C"],
    "Stand By Me (in C)": ["C", "C", "Am", "Am", "F", "G", "C", "C"],
    "Heart and Soul": ["C", "Am", "F", "G", "C", "Am", "F", "G"],
    "All 7 chords": ["C", "Dm", "Em", "F", "G", "Am", "Bdim", "C"],
}

NAME_TO_NUMBER = {name: n for n, (name, _) in CHORDS.items()}
