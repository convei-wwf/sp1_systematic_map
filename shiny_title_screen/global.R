### The app reads in a bibliography file and displays a title, author, year, journal, 
### and which search terms appear in the abstract, to give a little context.
### Criteria are displayed and the user can classify accordingly, then
### click "next" to move on to the next doc.

library(shiny)
library(shinyFiles)
library(tidyverse)
library(DT)
library(here)
library(synthesisr)



clean_bib <- function(bib_df, ext = '.ris') {
  if(nrow(bib_df) == 0) bib_df <- null_bib

  bib_clean_df <- bib_df %>%
    janitor::clean_names() %>%
    select(source_type, author, title, journal, year, abstract) %>%
    ### select the first author last name only
    mutate(author = str_remove_all(author, ',.+')) %>%
    # rename(first_author = author) %>%
    mutate(title = str_remove_all(title, '\\{|\\}')) %>%
    mutate(title = str_to_sentence(title),
           journal = str_to_title(journal),
           author = str_to_title(author))
    
  return(bib_clean_df)
    
}

null_bib <- data.frame(source_type = NA, author = NA, title = NA, 
                       journal = NA, year = NA, abstract = NA)

# bib_all <- read_refs(here('_data/output_for_colandr/sample.ris')) %>% clean_bib()

bib_screened_f <- here('_data/output_for_colandr/title_screened_cco.ris')
if(file.size(bib_screened_f) > 0 & file.exists(bib_screened_f)) {
  bib_screened <- read_refs(bib_screened_f) %>% clean_bib()
} else {
  bib_screened <- null_bib
}

bib_toscreen <- anti_join(bib_all, bib_screened)

### stitch a search term string
esi_terms <- 'satellite|space.based|remote observation|remote sensing|earth observation|remotely.sens[a-z]+|modis|landsat'
dec_terms <- 'decision|optimization|risk analysis|management|policy|cost.benefit analysis|benefit.cost analysis|investment|contingent valuation|counterfactual|value of information'
value_terms <- 'value|valuation|benefit|utility'
social_terms <- 'social|societal|cultural|([a-z]+-?)?economic|environmental|ecosystem service|sustainable development|protected area|heritage site|non.?use value'

search_terms <- paste(esi_terms, dec_terms, value_terms, social_terms, sep = '|')

embolden <- function(text, terms = search_terms) {
  indices <- str_locate_all(text, terms) 
  ### increase index of end positions:
  indices[[1]][ , 2] <- indices[[1]][ , 2] + 1
  ### set up as vector and go from the end to the start!
  i_vec <- indices %>% unlist() %>% sort(decreasing = TRUE)
  text_sub <- str_to_sentence(text)
  for(i in i_vec) {
    ### i <- 7
    stringi::stri_sub(text_sub, i, i-1) <- '**'
  }
  text_out <- markdown(text_sub) %>%
    str_replace_all('<strong>', '<strong style="color:#FF0000";>')
  return(text_out)
}

