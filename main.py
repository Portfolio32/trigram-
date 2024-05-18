import torch


words = open("names.txt").read().splitlines()

characters = list(sorted(set(''.join(words))))
characters.insert(0, '.')
character_pairs = []

for ch1 in characters:
    for ch2 in characters:
        character_pairs.append(ch1 + ch2)


stoi_pair = {s:i for i,s in enumerate(character_pairs)}
itos_pair = {s:i for i,s in stoi_pair.items()}
stoi = {s:i for i, s in enumerate(characters)}
itos = {s:i for i,s in stoi.items()}

N = torch.zeros((729,27), dtype=torch.int32)

for w in words: 
    w = ['.'] + list(w) + ['.']
    for ch1, ch2, ch3 in zip(w, w[1:], w[2:]):
        ch1ch2 = ch1 + ch2
        ix1 = stoi_pair[ch1ch2]
        ix2 = stoi[ch3]
        N[ix1, ix2] += 1



g = torch.Generator().manual_seed(2147483647)

p = N.float()
for i in range(10):

    out = []
    ix = torch.multinomial(p.sum(dim=1), num_samples=1,replacement=True, generator=g).item()
    out.append(itos_pair[ix])
    i = 0
    m = N[ix].float()
    ix = torch.multinomial(m, num_samples=1,replacement=True,generator=g).item()
    out.append(itos[ix])
    if ix == 0:
        pass
    else:
        i += 1
        ix = stoi_pair[out[i-1][1] + itos[ix]]
        while True:
            if ix == 0:
                break
            m = N[ix].float()
            ix = torch.multinomial(m, num_samples=1,replacement=True,generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
            i += 1
            ix = stoi_pair[out[i] + itos[ix]]
    print(''.join(out))



