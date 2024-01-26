load_bibtex <- function(dir = here('_data/bibtex_clean'), pattern, aspect = 'long') {
  
  ### NOTE: library(bib2df) ### use dev version: remotes::install_github("ropensci/bib2df")

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
  df <- data.table::fread(here('_data/results_clean_info.csv')) %>%
    oharac::dt_join(data.table::fread(here('_data/results_clean_text.csv')), 
                    by = 'doc_id', type = 'inner')
}

clean_author <- function(df) {
  c_names <- names(df)[names(df) != 'author']
  
  df1 <- df %>%
    unnest(author) %>%
    group_by(pick({{ c_names }})) %>%
    filter(author == first(author)) %>%
    ungroup() %>%
    mutate(author = tolower(author) %>%    ### drop caps
             stringi::stri_trans_general('Latin-ASCII') %>% ### drop diacritical marks
             str_remove(',.*') %>%         ### keep only last name
             str_remove_all('[^a-z ]') %>% ### drop punct
             str_squish()) %>%             ### drop whitespace
    distinct()
}

clean_text <- function(txt) {
  ### drop idiosyncratic punctuation:
  ### * diacritical marks
  ### * html tags
  ### * emdash -> dash; formatted quotes to unformatted; spaced-dash to no space
  ### * weird quote approximations: e.g., < ``carta della natura{''} >
  txt_out <- txt %>%
    tolower() %>%
    ### drop diacritical marks
    stringi::stri_trans_general('Latin-ASCII') %>%
    ### drop HTML tags using lazy regex
    str_remove_all('<.+?>') %>%
    ### replace funky non-standard punctuation
    str_replace_all('–|—', '-') %>%
    str_replace_all(' - ', '-') %>%
    ### fix quotes
    str_replace_all('“|”', '"') %>%
    str_replace_all("‘|’", "'") %>%
    str_replace_all('``|\\{\'\'(\\})?', '"') %>%
    ### escape backslashes and remaining curly braces
    str_remove_all('\\\\|\\{|\\}') %>%
    ### remove excess white space
    str_squish()
  
  return(txt_out)
}
