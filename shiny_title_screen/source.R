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

message('Loading source.R...')
library(shiny)
library(shinyFiles)
library(tidyverse)
library(DT)
library(here)
library(synthesisr)

### read bibtex given file selection (input$bibtex_fs) and action (input$load_bibtex)
f <- here('_data/output_for_colandr', 'sample.ris')

message('Loading refs from ', f)

docs_df <- read_file(f) 
begin <- docs_df %>% str_locate_all()


%>% parse_ris()

message('In full docs list, ', nrow(docs_df), ' documents found...')

### if a file of screened docs already exists, anti_join to the full docs list
### to just leave to-be-screened docs
bib_outf <- here('shiny_title_screen/app_out/title_screened.bib')
if(file.exists(bib_outf)) {
  screened <- bib2df::bib2df(bib_outf)
  message('Omitting ', nrow(screened), ' documents already screened...')
  
  docs_df <- docs_df %>%
    anti_join(screened %>% select(-EXTRA))
}
message('Returning, ', nrow(docs_df), ' documents to be screened...')

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

