"""Deterministic edit-local context for directional text pairs."""
from difflib import SequenceMatcher
import re


def segments(text, matches, keep):
    pieces = []
    start = None
    for i, include in enumerate([*keep,False]):
        if include and start is None:
            start = i
        elif not include and start is not None:
            pieces.append(text[matches[start].start():matches[i-1].end()])
            start = None
    return ' … '.join(pieces)


def edit_view(left, right):
    if left==right:
        return (left,right), False
    # Canonical orientation removes SequenceMatcher's directional tie-breaking.
    reverse = left>right
    pair = (right,left) if reverse else (left,right)
    matches = [list(re.finditer(r'\S+',s)) for s in pair]
    words = [[m.group().casefold() for m in line] for line in matches]
    keep = [[False]*len(line) for line in words]
    alignment = SequenceMatcher(None,*words,autojunk=False)
    for tag, a, b, c, d in alignment.get_opcodes():
        if tag=='equal':
            continue
        for mask,start,end in ((keep[0],a,b),(keep[1],c,d)):
            lo,hi = max(0,start-2),min(len(mask),end+2)
            mask[lo:hi] = [True]*(hi-lo)
    if min(map(sum,keep))<3 or sum(len(v)-sum(v) for v in keep)<4:
        return (left,right), False
    result = tuple(segments(s,m,k) for s,m,k in zip(pair,matches,keep,strict=True))
    return (result[::-1] if reverse else result), True
