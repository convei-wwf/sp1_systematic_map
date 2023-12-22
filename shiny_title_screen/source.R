#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#
### The app reads in a Bibtex file and displays a title, author, year, journal, 
### and which search terms appear in the abstract, to give a little context.
### Criteria are displayed and the user can classify accordingly, then
### click "next" to move on to the next doc.

library(shiny)
library(shinyFiles)
library(tidyverse)
library(DT)
library(here)

library(bib2df) ### use dev version: remotes::install_github("ropensci/bib2df")
if(packageVersion('bib2df') < '1.1.2.0') {
  ### should be version 1.1.2.0 or higher (1.1.1 is on CRAN)
  ### use dev version: remotes::install_github("ropensci/bib2df")
  stop('Package bib2df version: ', packageVersion(bib2df),
       '... Update bib2df from github: remotes::install_github("ropensci/bib2df"')
}

### read bibtex given file selection (input$bibtex_fs) and action (input$load_bibtex)
fs <- list.files(here('_data/bibtex_clean'),
                 # pattern = 'zot_benchmark_a.bib',
                 pattern = 'wos.bib|scopus.bib',
                 full.names = TRUE)
message('Loading bibtex from ', paste(basename(fs), collapse = ', '))
docs_df <- lapply(fs, bib2df::bib2df) %>%
  setNames(basename(fs)) %>%
  bind_rows(.id = 'bibtex_source') %>%
  distinct()
message('In full docs list, ', nrow(docs_df), ' documents found...')

### if a file of screened docs already exists, anti_join to the full docs list
### to just leave to-be-screened docs
bib_outf <- here('shiny_title_screen/app_out/title_screened.bib')
if(file.exists(bib_outf)) {
  screened <- bib2df::bib2df(bib_outf)
  message('Found, ', nrow(screened), ' documents already screened...')
  
  docs_df <- docs_df %>%
    anti_join(screened %>% select(-EXTRA))
}
message('Returning, ', nrow(docs_df), ' documents to be screened...')

### stitch a search term string
esi_terms <- 'satellite|space.based|remote observation|remote sensing|earth observation|remotely.sens[a-z]+|modis|landsat'
dec_terms <- 'decision|optimization|risk analysis|management|policy|cost.benefit analysis|benefit.cost analysis|investment|contingent valuation|counterfactual|value of information'
value_terms <- 'value|valuation|benefit|utility'
social_terms <- 'social|societal|cultural|[a-z]+-?economic|environmental|ecosystem service|sustainable development|protected area|heritage site|non.?use value'

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
  text_out <- markdown(text_sub)
  return(text_out)
}

