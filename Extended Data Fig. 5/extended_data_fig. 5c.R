library(tidyverse)
library(viridis)

# import data
data <- read.table("data_stat-for_plot.txt", header = TRUE, sep = "\t")
# omics_level <- colnames(data)[-c(1,2)]
omics_level <- rev(colnames(data)[-c(1,2)])
species_level <- data$Species
data <- data %>% gather(key = "observation", value="value", -c(1,2))
data$observation <- factor(data$observation, levels = omics_level)
data$Species <- factor(data$Species, levels = species_level)
# 再每组末尾添加空白距离
empty_bar <- 2
nObsType <- nlevels(as.factor(data$observation))
to_add <- data.frame( matrix(NA, empty_bar*nlevels(data$group)*nObsType, ncol(data)) )
colnames(to_add) <- colnames(data)
to_add$group <- rep(levels(data$group), each=empty_bar*nObsType )
data <- rbind(data, to_add)
data <- data %>% arrange(Group, Species)
data$id <- rep( seq(1, nrow(data)/nObsType) , each=nObsType)

#获取每个标签的名称和y轴的位置
label_data <- data %>% group_by(id, Species) %>% summarize(tot=sum(value))
number_of_bar <- nrow(label_data)
angle <- 90 - 360 * (label_data$id-0.5) /number_of_bar     # I substract 0.5 because the letter must have the angle of the center of the bars. Not extreme right(1) or extreme left (0)
label_data$hjust <- ifelse( angle < -90, 1, 0)
label_data$angle <- ifelse(angle < -90, angle+180, angle)

# 为基线准备一个数据帧
# base_data <- data %>%
#   group_by(Group) %>%
#   summarize(start=min(id), end=max(id) - empty_bar) %>%
#   rowwise() %>%
#   mutate(title=mean(c(start, end)))
#
# # 为grid准备一个数据帧
# grid_data <- base_data
# grid_data$end <- grid_data$end[ c( nrow(grid_data), 1:nrow(grid_data)-1)] + 1
# grid_data$start <- grid_data$start - 1
# grid_data <- grid_data[-1,]

# 绘图
p <- ggplot(data) +
  # 添加堆叠的条形图
  geom_bar(aes(x=as.factor(id), y=value, fill=observation), stat="identity", alpha=0.5) +
  scale_fill_viridis(discrete=TRUE) +

  # Add a val=100/75/50/25 lines.
  # geom_segment(data=grid_data, aes(x = end, y = 0, xend = start, yend = 0), colour = "grey", alpha=1, size=0.3 , inherit.aes = FALSE ) +
  # geom_segment(data=grid_data, aes(x = end, y = 50, xend = start, yend = 50), colour = "grey", alpha=1, size=0.3 , inherit.aes = FALSE ) +
  # geom_segment(data=grid_data, aes(x = end, y = 100, xend = start, yend = 100), colour = "grey", alpha=1, size=0.3 , inherit.aes = FALSE ) +
  # geom_segment(data=grid_data, aes(x = end, y = 150, xend = start, yend = 150), colour = "grey", alpha=1, size=0.3 , inherit.aes = FALSE ) +
  # geom_segment(data=grid_data, aes(x = end, y = 200, xend = start, yend = 200), colour = "grey", alpha=1, size=0.3 , inherit.aes = FALSE ) +
  #
  # 添加显示100/75/50/25的文本
  # ggplot2::annotate("text", x = rep(max(data$id),5), y = c(0, 50, 100, 150, 200), label = c("0", "50", "100", "150", "200") , color="grey", size=6 , angle=0, fontface="bold", hjust=1) +

  ylim(-15,max(label_data$tot, na.rm=T)) +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(rep(-1,4), "cm")
  ) +
  # coord_polar() +
  # geom_text(data=label_data, aes(x=id, y=tot, label=Species, hjust=hjust), color="black", fontface="bold",alpha=0.6, size=5, angle= label_data$angle, inherit.aes = FALSE )
  coord_polar()

# 在每个条形图顶部添加一个标签
# geom_text(data=label_data, aes(x=id, y=tot+10, label=individual, hjust=hjust), color="black", fontface="bold",alpha=0.6, size=5, angle= label_data$angle, inherit.aes = FALSE ) +

# 添加基线信息
# geom_segment(data=base_data, aes(x = start, y = -5, xend = end, yend = -5), colour = "black", alpha=0.8, size=0.6 , inherit.aes = FALSE )  +
# geom_text(data=base_data, aes(x = title, y = -18, label=group), hjust=c(1,1,0,0), colour = "black", alpha=0.8, size=4, fontface="bold", inherit.aes = FALSE)

p
ggsave(p, file="extended_data_fig.5c.png", width=12, height=12)
