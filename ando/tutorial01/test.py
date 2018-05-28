from collections import defaultdict

breakfast = ['kome', 'pan', 'kome', 'kome', 'cereal', 'pan', 'kome']

numbers = defaultdict(lambda: 0)
for b in breakfast:
    numbers[b] += 1
