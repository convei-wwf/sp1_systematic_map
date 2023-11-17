load_articles <- function(dir = here('_data/bibtex_clean'), pattern = 'wos_', aspect = 'long') {
  bib_clean_fs <- list.files(path = dir, pattern = pattern, full.names = TRUE)

  all_fields_df <- lapply(bib_clean_fs, bib2df::bib2df) %>%
    bind_rows() %>%
    janitor::clean_names()
  
  if(aspect == 'long') {
    topic_df <- all_fields_df %>%
      select(author, journal, year, title, 
             abstract, keywords, keywords_plus, 
             web_of_science_categories) %>%
      mutate(title_protect = title) %>%
      pivot_longer(cols = c(title, abstract, keywords, keywords_plus), names_to = 'topic', values_to = 'text') %>%
      rename(title = title_protect) %>%
      mutate(text = tolower(text)) %>%
      filter(!is.na(text)) %>%
      distinct()
    
    return(topic_df)
  }
  if(aspect == 'wide') {
    return(all_fields_df)
  }
  stop('Argument `aspect` must be either "wide" or "long" aspect ratio!')
}
