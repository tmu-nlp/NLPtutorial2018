library(ggplot2)

args = commandArgs(T)
if(length(args) != 1)
    stop("Please provide a tsv file in this folder!")

data = read.delim(args[1])

p <- ggplot(data, aes(reorder(paste(id, stringr::str_wrap(seq, 10), sep = "\n"), id), postag, fill = cost)) +
    geom_bin2d( stat = "identity", show.legend = F) +
    #scale_fill_gradientn(colours = terrain.colors(10)) +
    scale_fill_gradient(low = "white", high = "black") +
    labs( x = 'sentence', y = "POS tags", title = "Cost")

ggsave(paste(args[1], ".png", sep = ""), width = nrow(data) / 70, height = 10)
