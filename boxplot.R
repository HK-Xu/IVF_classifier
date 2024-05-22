suppressMessages(library(readr))
suppressMessages(library(dplyr))
suppressMessages(library(ggpubr))

windowsFonts(Times=windowsFont("Times New Roman"))

quant_data_path <- ""
group_info_path <- ""

# input data and pre-process
group_info <- read_delim(group_info_path, delim = "\t", show_col_types = FALSE) %>%
  dplyr::select(Sample, Group) 

plot_data <- read_delim(quant_data_path, delim = "\t", show_col_types = FALSE) %>% 
  tidyr::pivot_longer(cols = -Sample, names_to = "ID", values_to = "Intensity") %>%
  merge(group_info, by.x = "Sample", by.y = "Sample", all.y = TRUE) 

# set comparison groups
comparisons <- list(c("A", "B"), c("A", "C"), c("B", "C"))
# comparisons <- list(c("A", "B"))

ggboxplot(plot_data, x = "Group", y = "Intensity", color = "Group", bxp.errorbar = T, facet.by = "ID", scales = "free_y", ncol = 6, shrink = FALSE, outlier.size = 0.5, outlier.shape = NA) +
  stat_compare_means(comparisons = comparisons, method = "t.test" , label.x = 1.7, vjust = 0, size = 2.5, family = "Times", colour = "black") + 
  geom_point(position = position_jitter(width = 0.05), size = 0.2) +
  theme(text = element_text(family = "Times", colour = "black"),
        strip.text = element_text(size = 10, face = "bold", family = "Times"), #size = rel(1)
        strip.background = element_rect(fill = "lightblue", colour = "black", linewidth = 0.5),
        strip.placement = "top", 
        legend.title = element_text(size = 12, face = "bold", family = "Times"),
        legend.text = element_text(size = 10, lineheight = 0.5, family = "Times"), 
        legend.key.height = unit(0.7, "cm"),
        axis.title.x = element_blank(),
        axis.title.y = element_text(size = 12, face = "bold", family = "Times"),
        axis.text = element_text(angle = 0, vjust = 0, hjust = 0.5, size = 8, family = "Times"))
ggsave(filename = "/Features_boxPlot.png", device = "png", width = 10, height = 8, dpi = 300)
ggsave(filename = "/Features_boxPlot.pdf", device = "pdf", width = 10, height = 8) 
