load_bibtex <- function(dir = here('_data/bibtex_clean'), pattern = 'wos_', aspect = 'long') {
  bib_clean_fs <- list.files(path = dir, pattern = pattern, full.names = TRUE)

  
  all_fields_df <- lapply(bib_clean_fs, bib2df::bib2df) %>%
    lapply(janitor::clean_names) %>%
    setNames(basename(bib_clean_fs)) %>%
    bind_rows(.id = 'bibtex_source') %>%
    distinct()
  
  if(aspect == 'long') {
    field_df <- all_fields_df %>%
      select(author, journal, year, title, doi,
             abstract, contains('keyword'),
             web_of_science_categories,
             bibtex_source) %>%
      distinct() %>%
      mutate(title_protect = title) %>%
      pivot_longer(cols = c(title, abstract, web_of_science_categories, contains('keyword')), 
                   names_to = 'field', values_to = 'text') %>%
      rename(title = title_protect) %>%
      mutate(text = tolower(text)) %>%
      filter(!is.na(text)) %>%
      distinct()
    
    return(field_df)
  }
  if(aspect == 'wide') {
    return(all_fields_df)
  }
  stop('Argument `aspect` must be either "wide" or "long" aspect ratio!')
}

load_articles <- function() {
  data.table::fread(here('_data/results_clean.csv'))
}
