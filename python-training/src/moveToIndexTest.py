OBS_DIM = 768
PROMO_V = 5
N_SQ = 64
ACTION_DIM = N_SQ * N_SQ #64*64, promotions are handled separately through the promo handler


""""
Old version:
def moveToIndex(m: pb.ProtoMove):
    return (int(m.from_sq) * N_SQ) + int(m.to_sq)

def indextoMove(index, promo_masking):
    to_sq = index%N_SQ
    from_sq = ((index-to_sq)/N_SQ)%N_SQ
    return pb.ProtoMove(from_sq=from_sq, to_sq=to_sq, promotion=promo_masking)
    
    
Working ProtoMove version:
def moveToIndex(m: pb.ProtoMove):
    if m.promotion != 0:
        index = 4095 + ((m.from_sq%8)*2 + m.to_sq)*4 + m.promotion
    else:
        index = (m.from_sq * N_SQ) + m.to_sq

    return index

def indexToMove(index):
    if index > 4183:
        raise Exception("Index out of range")

    if index > 4095: #promotion move
        rest = index - 4095
        promotion = rest%4
        if promotion == 0:
            promotion = 4
        rest2 = (rest-promotion)/4
        from_h = int((rest2+1)/3)
        from_sq = 8+from_h
        to_sq = rest2-from_h*2
    else:
        to_sq = index%N_SQ
        from_sq = ((index-to_sq)/N_SQ)%N_SQ
        promotion = 0
    return pb.ProtoMove(from_sq=from_sq, to_sq=to_sq, promotion=promotion)
    
"""

def moveToIndex(from_sq, to_sq, promotion):
    if promotion != 0:
        index = 4095 + ((from_sq%8)*2 + to_sq)*4 + promotion
    else:
        index = (from_sq * N_SQ) + to_sq

    return int(index)

def indexToMove(index):
    if index > 4095: #promotion move
        rest = index - 4095
        promotion = rest%4
        if promotion == 0:
            promotion = 4
        rest2 = (rest-promotion)/4
        from_h = int((rest2+1)/3)
        from_sq = 8+from_h
        to_sq = rest2-from_h*2
    else:
        to_sq = index%N_SQ
        from_sq = ((index-to_sq)/N_SQ)%N_SQ
        promotion = 0
    return (int(from_sq), int(to_sq), int(promotion))

def testAll():
    for i in range(8):
        for p in range(1,5):
            u1 = i-1
            u2 = i+1
            if u1>=0:
                index = moveToIndex(8+i, u1, p)
                (a, b, c) = indexToMove(index)
                print(index)
                if a!=8+i or b !=u1 or c != p:
                    print("bad: " + str(index))
            if u2 < 8:
                index2 = moveToIndex(8+i, u2, p)
                (a2, b2, c2) = indexToMove(index2)
                print(index2)
                if a2!=8+i or b2 !=u2 or c2 != p:
                    print("bad" + str(index2))

            index3 = moveToIndex(8+i, i, p)
            (a3, b3, c3) = indexToMove(index3)
            print(index3)
            if a3!=8+i or b3 !=i or c3 != p:
                print("bad" + str(index3))

if __name__=="__main__":
    testAll()

    print("good")

