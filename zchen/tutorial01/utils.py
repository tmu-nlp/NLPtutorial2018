s_tok = "s"

def n_gram(n, s):
    ns = []
    for i in range(n):
        ns.append(s[i:])
    return tuple(zip(*ns))

def make_padding(n):
    # this maker is xml. so single ending should appended
    sos = ["<%s>"  % s_tok] * (n)
    eos = ["</%s>" % s_tok]
    return sos, eos

def count_n_gram_from_file(n, fr):
    from collections import defaultdict
    count_dict = defaultdict(lambda : 0)
    # make padding
    sos, eos = make_padding( n-1 )

    for line in fr:
        # apply padding for list of tokens
        lot = line.strip().split(" ")
        lot = sos + lot + eos if s_tok else lot
        for token in n_gram(n, lot):
            count_dict[token] += 1
    return count_dict

def seal_model(count_dict):
    total_count = sum(count_dict.values())
    return total_count, { key: value/total_count for key, value in count_dict.items()}

if __name__ == "__main__":
    print(n_gram(1, "I am an NLPer"))
    print(n_gram(2, "I am an NLPer"))

    import sys
    with open(sys.argv[1], 'r') as fr:
        count_dict = count_n_gram_from_file(2, fr)
        for pair in seal_model(count_dict, True).items():
            print( *pair )
