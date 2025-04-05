# Precompute all possible 8-bit binary strings (0-255)
D2B_LOOKUP = [f"{i:08b}" for i in range(256)]

def HA(X, Y):

    bit_sum = np.array([[0,1], [1,0]])
    bit_carry = np.array([[0,0], [0,1]])

    return bit_carry[X][Y], bit_sum[X][Y]


def FA(x,y,z):
    c1,s1 = HA(x,y)
    c2,s2 = HA(s1,z)
    carry = c1 + c2
    return carry,s2



def D2B(pxl):
    """Ultra-fast decimal to binary using lookup table"""
    return D2B_LOOKUP[pxl]



def decimal(n):
    k = list(n)
    sum = 0
    k.reverse()
    for i in range(len(k)):
        sum = sum + (int(k[i]) * (2**i))
    return sum

def  andg(a,b):
     z = a * b
     return z

def TER_CONC4(a,b,c,d):
    Y=str(a)+str(b)+str(c)+str(d)
    V=decimal(Y)
    return V
def cmp(p,q,r,s,t):
    c1,s1 = FA(p,q,r)
    carry,sum = FA(s1,s,t)
    return c1,carry,sum



### Fully Approximation
def Binary_Multiply8_full_approx(m, n):
      a = D2B(m)
      b = D2B(n)


      p77 = andg(int(a[7]),int(b[7]))
      p67 = andg(int(a[6]),int(b[7]))
      p57 = andg(int(a[5]),int(b[7]))
      p47 = andg(int(a[4]),int(b[7]))
      p37 = andg(int(a[3]),int(b[7]))
      p27 = andg(int(a[2]),int(b[7]))
      p17 = andg(int(a[1]),int(b[7]))
      p07 = andg(int(a[0]),int(b[7]))

      p76 = andg(int(a[7]),int(b[6]))
      p66 = andg(int(a[6]),int(b[6]))
      p56 = andg(int(a[5]),int(b[6]))
      p46 = andg(int(a[4]),int(b[6]))
      p36 = andg(int(a[3]),int(b[6]))
      p26 = andg(int(a[2]),int(b[6]))
      p16 = andg(int(a[1]),int(b[6]))
      p06 = andg(int(a[0]),int(b[6]))

      p75 = andg(int(a[7]),int(b[5]))
      p65 = andg(int(a[6]),int(b[5]))
      p55 = andg(int(a[5]),int(b[5]))
      p45 = andg(int(a[4]),int(b[5]))
      p35 = andg(int(a[3]),int(b[5]))
      p25 = andg(int(a[2]),int(b[5]))
      p15 = andg(int(a[1]),int(b[5]))
      p05 = andg(int(a[0]),int(b[5]))


      p74 = andg(int(a[7]),int(b[4]))
      p64 = andg(int(a[6]),int(b[4]))
      p54 = andg(int(a[5]),int(b[4]))
      p44 = andg(int(a[4]),int(b[4]))
      p34 = andg(int(a[3]),int(b[4]))
      p24 = andg(int(a[2]),int(b[4]))
      p14 = andg(int(a[1]),int(b[4]))
      p04 = andg(int(a[0]),int(b[4]))

      p73 = andg(int(a[7]),int(b[3]))
      p63 = andg(int(a[6]),int(b[3]))
      p53 = andg(int(a[5]),int(b[3]))
      p43 = andg(int(a[4]),int(b[3]))
      p33 = andg(int(a[3]),int(b[3]))
      p23 = andg(int(a[2]),int(b[3]))
      p13 = andg(int(a[1]),int(b[3]))
      p03 = andg(int(a[0]),int(b[3]))


      p72 = andg(int(a[7]),int(b[2]))
      p62 = andg(int(a[6]),int(b[2]))
      p52 = andg(int(a[5]),int(b[2]))
      p42 = andg(int(a[4]),int(b[2]))
      p32 = andg(int(a[3]),int(b[2]))
      p22 = andg(int(a[2]),int(b[2]))
      p12 = andg(int(a[1]),int(b[2]))
      p02 = andg(int(a[0]),int(b[2]))

      p71 = andg(int(a[7]),int(b[1]))
      p61 = andg(int(a[6]),int(b[1]))
      p51 = andg(int(a[5]),int(b[1]))
      p41 = andg(int(a[4]),int(b[1]))
      p31 = andg(int(a[3]),int(b[1]))
      p21 = andg(int(a[2]),int(b[1]))
      p11 = andg(int(a[1]),int(b[1]))
      p01 = andg(int(a[0]),int(b[1]))


      p70 = andg(int(a[7]),int(b[0]))
      p60 = andg(int(a[6]),int(b[0]))
      p50 = andg(int(a[5]),int(b[0]))
      p40 = andg(int(a[4]),int(b[0]))
      p30 = andg(int(a[3]),int(b[0]))
      p20 = andg(int(a[2]),int(b[0]))
      p10 = andg(int(a[1]),int(b[0]))
      p00 = andg(int(a[0]),int(b[0]))

            # FIRST STAGE

      hc1,hs1 = HA(p37,p46)
      cc1,cs1 = cmp(p27,p36,p45,p54)
      cc2,cs2 = cmp(p17,p26,p35,p44)
      hc2,hs2 = HA(p53,p62)

      cc3,cs3 = cmp(p07,p16,p25,p34)
      cc4,cs4 = cmp(p43,p52,p61,p70)
      cc5,cs5 = cmp(p06,p15,p24,p33)
      fc1,fs1 = FA(p42,p51,p60)

      cc6,cs6 = cmp(p05,p14,p32,p23)
      hc3,hs3 = HA(p41,p50)

      cc7,cs7 = cmp(p04,p13,p22,p31)
      hc4,hs4 = HA(p03,p12)

          # second stage

      hc5,hs5 = HA(p57,p66)

      cc8,cs8 =   cmp(p47,p56,p65,p74)
      cc9,cs9 =   cmp(hs1,p55,p64,p73)
      cc10,cs10=  cmp(cs1,hc1,p63,p72)
      cc11,cs11 = cmp(cs2,hs2,cc1,p71)

      cc12,cs12 = cmp(cc2,hc2,cs3,cs4)
      cc13,cs13 = cmp(cc3,cc4,cs5,fs1)
      cc14,cs14=  cmp(fc1,cc5,cs6,hs3)
      cc15,cs15 = cmp(hc3,cc6,cs7,p40)
      cc16,cs16=  cmp(cc7,hs4,p21,p30)
      cc17,cs17 = cmp(hc4,p02,p11,p20)

      hc6,hs6 = HA(p01,p10)


             # third stage

      hc7,hs7 = HA(p67,p76)

      fc2,fs2 = FA(hs5,p75,hc7)

      fc3,fs3 = FA(cs8,hc5,fc2)
      fc4,fs4 = FA(cs9,cc8,fc3)

      fc5,fs5 = FA(cs10,cc9,fc4)
      fc6,fs6 = FA(cs11,cc10,fc5)
      fc7,fs7 = FA(cs12,cc11,fc6)
      fc8,fs8 = FA(cs13,cc12,fc7)
      fc9,fs9 = FA(cs14,cc13,fc8)
      fc10,fs10 = FA(cs15,cc14,fc9)
      fc11,fs11 = FA(cs16,cc15,fc10)
      fc12,fs12 = FA(cs17,cc16,fc11)
      fc13,fs13 = FA(hs6,cc17,fc12)
      fc14,fs14 = FA(p00,hc6,fc13)



      y = fc14,fs14,fs13,fs12,fs11,fs10,fs9,fs8,fs7,fs6,fs5,fs4,fs3,fs2,hs7,p77
      z=decimal(y)
      return z

#proposed
# def cmp(a,b,c,d):
#     S=[0,1,1,0, 1,0,0,1, 1,0,0,1, 0,1,1,1];

#     C=[0,0,0,1, 0,1,1,1, 0,1,1,1, 1,1,1,1];

#     V=TER_CONC4(a,b,c,d)

#     SUM = S[V]
#     CARRY= C[V]

#     return [CARRY,SUM]

## CAAM

def cmp(a,b,c,d):
    S=[0,1,1,1, 1,0,0,0, 1,0,0,0, 0,1,1,1];

    C=[0,0,0,0, 0,1,1,1, 0,1,1,1, 1,1,1,1];


    V=TER_CONC4(a,b,c,d)

    SUM = S[V]
    CARRY= C[V]

    return [CARRY,SUM]
