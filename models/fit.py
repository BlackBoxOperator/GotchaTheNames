#!/bin/python
import sys
from pprint import pprint

def range2str(rng):
    klass, tot, *rest = rng
    if klass == 2:
        return 'threshold == {:.3f}'.format(rest[0], rest[0] * tot)
    elif klass == 1:
        x, v = rest
        if x > v:
            x, v = v, x
            return '{:.3f} ({}) <= t < {:.3f} ({})'.format(
                    x, x * tot, v, v * tot)
        else:
            return '{:.3f} ({}) < t <= {:.3f} ({})'.format(
                    x, x * tot, v, v * tot)
    elif klass == 0:
        x, v = rest
        if x > v:
            x, v = v, x
            return '{:.3f} ({:3f}) < t <= {:.3f} ({:3f})'.format(
                 x, x * tot, v, v * tot)
        else:
            return '{:.3f} ({:3f}) <= t < {:.3f} ({:3f})'.format(
                 x, x * tot, v, v * tot)
    else: return 'wrong format'

def fitness_data(v, x, total):
    v.sort(), x.sort()
    acc, vidx, xidx = 0, 0, 0
    vlen , xlen = len(v), len(x)
    tlen = vlen + xlen

    threshold, bvidx, bxidx = 0, 0, 0

    while vidx < vlen or xidx < xlen:
        if vidx != vlen - 1 and (xidx == xlen - 1 or v[vidx] < x[xidx]):
            if xidx == xlen - 1:
                nacc = max(acc, (xidx + 1 + (vlen - vidx)) / tlen)
                nrng = (1, total, x[max(xidx - 1, 0)], v[vidx])
            else:
                nacc = max(acc, (xidx + (vlen - vidx)) / tlen)
                nrng = (0, total, x[max(xidx - 1, 0)], v[vidx])
            if nacc > acc:
                threshold = (x[max(xidx - 1, 0)] + v[vidx]) / 2
                acc, rng, bvidx, bxidx = nacc, nrng, vidx, xidx
            vidx += 1
        elif xidx != xlen - 1 and (vidx == vlen - 1 or v[vidx] > x[xidx]):
            if vidx == vlen - 1:
                nacc = max(acc, (min((xidx + 1), xlen) + (vlen - vidx)) / (vlen + xlen))
                nrng = (0, total, x[xidx], v[vidx])
            else:
                nacc = max(acc, (min((xidx + 1), xlen) + (vlen - vidx)) / (vlen + xlen))
                nrng = (1, total, x[xidx], v[vidx])
            if nacc > acc:
                threshold = (v[vidx] + x[xidx]) / 2
                acc, rng, bvidx, bxidx = nacc, nrng, vidx, xidx
            xidx += 1
        else:
            nacc = max(acc, (xidx + (vlen - vidx) - 1) / (vlen + xlen))
            nrng = (2, total, x[xidx])
            if nacc > acc:
                threshold = v[vidx]
                acc, rng, bvidx, bxidx = nacc, nrng, vidx, xidx
            vidx += 1
            xidx += 1

    vacc = len([val for val in v if val > threshold])
    verr = vlen - vacc
    xacc = len([val for val in x if val <= threshold])
    xerr = xlen - xacc
    acc = (vacc + xacc) / tlen
    return threshold, rng, acc, vacc, verr, vlen, xacc, xerr, xlen

def evaluate(ban, d, total):

    print("{} {:10s} {}".format("=" * 10, ban, "=" * 10))

    v, x = d["v"], d["x"]
    threshold, rng, acc, vacc, verr, vlen, xacc, xerr, xlen = fitness_data(v, x, total)

    print('acc: {:.3f}'.format(acc))
    print('aml acc     = {:4d} / {:4d} ({:.5f})'.format(vacc, vlen, vacc / vlen))
    print('aml miss    = {:4d} / {:4d} ({:.5f})'.format(verr, vlen, verr / vlen))
    print('non-aml acc = {:4d} / {:4d} ({:.5f})'.format(xacc, xlen, xacc / xlen))
    print('non-aml err = {:4d} / {:4d} ({:.5f})'.format(xerr, xlen, xerr / xlen))
    print('range : {}'.format(range2str(rng)))
    return threshold

def fitness(v, x, total):
    """
    v: aml
    x: non-aml
    return overall_acc, true positive, false negative
    """
    threshold, rng, acc, vacc, verr, vlen, xacc, xerr, xlen = fitness_data(v, x, total)
    return acc, vacc / vlen, xacc / xlen

if __name__ == '__main__':
    count = {'v': [], 'x': []}
    totsim = {'v': [], 'x': []}
    avgsim = {'v': [], 'x': []}

    avg_list = []

    for row in open(sys.argv[1]).read().split('\n'):
        if not row.strip(): continue
        _, label, good, _, total, _, _, _, _, score, _, *rest =  row.split()
        count[label].append(int(good) / int(total))
        totsim[label].append(float(score))
        avgsim[label].append(float(score) / int(total))
        avg_list.append((float(score) / int(total), label, ''.join(rest)))

    evaluate("count", count, float(total))
    evaluate("AML similarity sum", totsim, float(total))
    thres = evaluate("AML similarity avg", avgsim, float(total))
    print("aml miss:")
    pprint([(s, cap) for s, l, cap in avg_list if l == 'v' and s < thres])

    print("non-aml error:")
    pprint([(s, cap) for s, l, cap in avg_list if l == 'x' and s >= thres])
