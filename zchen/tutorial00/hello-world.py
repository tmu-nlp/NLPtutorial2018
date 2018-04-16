print("Hello World!")

my_int      = 4
my_float    = 2.3
my_string   = 'one'
fmt_old     = "string: %s \t float: %f \t int: %d"
data        = (my_string, my_float, my_int)
fmt_new     = "{2}, {0}, {1}"
# new format string relates to __format__(self, spec)
# extract returns with spec in {:spec} form.

print(fmt_old % data)
print(fmt_new.format(*data))
