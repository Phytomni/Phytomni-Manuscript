library(tidyverse)
library(scales)
library(treemapify)

plot_f_s1A <- function(data_path = "./figure_s1/paper_meta_year_counts.csv",
                       output_path = "./figure_s1/figure_s1_A.pdf") {
  
  df_full <- read_csv(data_path) %>%
    right_join(tibble(PY = 1901:2025), by = "PY") %>%
    mutate(count = replace_na(count, 0)) %>%
    arrange(PY) %>%
    mutate(cum_count = cumsum(count))
  
  df_bin_cumulative <- df_full %>%
    mutate(
      interval_start = if_else(PY <= 1980, 
                               20 * floor((PY - 1901) / 20) + 1901,
                               5 * floor((PY - 1981) / 5) + 1981),
      interval_end = if_else(PY <= 1980, interval_start + 19, interval_start + 4)
    ) %>%
    filter(PY == interval_end) %>%
    mutate(year_label = str_c(interval_start, interval_end, sep = "-"))
  
  p <- ggplot(df_bin_cumulative, aes(year_label, cum_count, group = 1)) +
    geom_line(color = "#0072B2", linewidth = 1) +
    geom_point(color = "#0072B2", size = 2) +
    scale_x_discrete(guide = guide_axis(angle = 45)) +
    scale_y_continuous(
      labels = number_format(big.mark = " "),
      expand = expansion(mult = c(0, 0.05))
    ) +
    labs(
      title = "Cumulative Count of Papers (1901-2025)",
      x = "Time Interval",
      y = "Cumulative Count"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
      plot.background = element_rect(color = "black", fill = NA, linewidth = 1),
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.title = element_text(face = "bold"),
      panel.grid.major = element_line(linetype = "dashed", color = "grey80")
    )
  
  ggsave(output_path, p, width = 8, height = 5, device = cairo_pdf)
}

plot_f_s1B <- function(data_path = "./figure_s1/type_distribution.csv",
                       output_path = "./figure_s1/figure_s1_B.pdf",
                       use_col = "SC_first",
                       top_n = 30) {
  
  merged_df <- read_csv(data_path)
  
  processed_df <- merged_df %>%
    rename(analysis_col = all_of(use_col)) %>%
    mutate(analysis_col = str_remove(analysis_col, " - Other Topics")) %>%
    filter(!is.na(analysis_col) & analysis_col != "")
  
  type_counts <- processed_df %>%
    count(analysis_col, name = "n") %>%
    arrange(desc(n))
  
  if (nrow(type_counts) > top_n) {
    top_counts <- type_counts[1:top_n, ]
    other_count <- tibble(analysis_col = "Others", 
                          n = sum(type_counts$n[(top_n+1):nrow(type_counts)]))
    type_counts <- bind_rows(top_counts, other_count)
  }
  
  type_counts <- type_counts %>%
    mutate(percentage = n / sum(n))
  
  p <- ggplot(type_counts, aes(area = n, fill = analysis_col)) +
    geom_treemap(color = "white", size = 1.5, alpha = 0.9) +
    scale_fill_manual(
      values = colorRampPalette(brewer.pal(12, "Set3"))(nrow(type_counts)),
      guide = guide_legend(ncol = ifelse(nrow(type_counts) > 30, 2, 1))
    ) +
    labs(
      title = "Document Type Distribution",
      subtitle = str_glue(
        "Total categories: {nrow(processed_df)} | Displayed: {nrow(type_counts)}"
      )
    ) +
    theme_void() +
    theme(
      plot.title = element_text(size = 24, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 16, hjust = 0.5),
      legend.position = "bottom",
      legend.title = element_text(face = "bold")
    )
  
  ggsave(output_path, p, width = 5, height = 12)
}

##################
plot_f_s1A()
plot_f_s1B()


