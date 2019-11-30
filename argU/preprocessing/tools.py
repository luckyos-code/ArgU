import re


def denoise(text):
    """Entferne Rauschen aus einem Text

    1. Lower Case
    2. Alles in eckigen Klammern wird entfernt
    3. Entferne Web-Adressen
    4. Entferne Sonderzeichen außer Apostrophe

    Returns:
        list mit den einzelnen, aufeinanderfolgenden Wörtern
    """

    denoised_text = []

    text = re.sub(r'\[[^]]*\]', '', text)
    for word in text.split():
        word = word.lower()
        if not any(w in word for w in ['www.', '.com', 'com/']):
            word = re.sub(r'[^\w\s]', ' ', word).strip()
            word_split = word.split()
            if len(word_split) > 1:
                if word_split[0] == 'u' and word_split[1] == 's':
                    # USA in Großbuchstaben, um es von 'us' zu unterscheiden
                    denoised_text.append('US')
                elif word_split[1] in ['s', 'nt', 't']:
                    # Wörter mit Apostroph
                    denoised_text.append('\''.join(word_split))
                else:
                    # Rest
                    denoised_text.extend(word_split)
            elif word != '':
                denoised_text.append(word)

    return denoised_text
