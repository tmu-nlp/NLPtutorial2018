import argparse

def arguments_parse():
    parser = argparse.ArgumentParser(
        usage='テキストファイルから「単語の異なり数」および「頻度」を表示する',
        description='説明',
        epilog='end',
        add_help=True,
    )
    parser.add_argument('file_name', help='入力ファイル名', type=str)
    parser.add_argument('-m', '--mode', help='異なり数非表示:0 異なり数表示:1 省略時は表示', type=int, choices=[0,1])
    parser.add_argument('-o', '--order', help='名前昇順:0 名前降順:1 値降順:2 値昇順:3 省略時は名前順', type=int, choices=[0,1,2,3])
    parser.add_argument('-n', '--number', help='上位n件の表示，省略時は全件表示', type=int)
    return parser.parse_args()

if __name__ == '__main__':

    args = arguments_parse()

    my_file = open(args.file_name, "r", encoding="utf-8")

    countdict = dict()
    for line in my_file:
        line = line.strip()
        words = line.split(" ")
        for word in words:
            if word in countdict.keys():
                countdict[word] += 1
            else:
                countdict[word] = 1
    
    if args.mode != 0:
        print('単語の異なり数')
        print(len(countdict))
        print('---')

    print('数単語の頻度')
    if  not args.order:
        sortedlist = sorted(countdict.items())
    elif args.order == 1:
        sortedlist = sorted(countdict.items(), reverse=True)
    elif args.order == 2:
        sortedlist = sorted(countdict.items(), key=lambda x:x[1], reverse=True)
    elif args.order == 3:
        sortedlist = sorted(countdict.items(), key=lambda x:x[1])
    else:
        sortedlist = []

    for i, (key, value) in enumerate(sortedlist):
        if args.number is not None and args.number <= i:
            break
        print(f'{key} {value}')
