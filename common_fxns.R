load_articles <- function(dir = here('bibtex_clean'), pattern = 'wos_') {
  bib_clean_fs <- list.files(path = dir, pattern = pattern, full.names = TRUE)

  all_fields_df <- lapply(bib_clean_fs, bib2df::bib2df) %>%
    bind_rows() %>%
    janitor::clean_names()
  
  topic_df <- all_fields_df %>%
    select(author, journal, year, title, abstract, keywords, keywords_plus, web_of_science_categories) %>%
    mutate(title_protect = title) %>%
    pivot_longer(cols = c(title, abstract, keywords, keywords_plus), names_to = 'topic', values_to = 'text') %>%
    rename(title = title_protect) %>%
    mutate(text = tolower(text)) %>%
    filter(!is.na(text)) %>%
    distinct()
  
  return(topic_df)
}
