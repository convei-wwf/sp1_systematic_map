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

### Checkboxes for criteria:
### * Explicitly mentions satellite or earth obs in title
### * Explicitly mentions value/benefit/utility with respect to EO data
### * Explicitly mentions a societal benefit
### Radio buttons for categorization
### * Definitely in scope
### * Maybe (keep)
### * Definitely out of scope
### Action button to store values and select new title
### * Append record to an output bibtex file with categorization in 'extra' field
### * How to keep track of uncategorized vs categorized - maybe on startup:
###    * read in full records
###    * read in categorized records
###    * anti-join categorized records to full records, dropping the 'extra' field (since that changes with categorization)
###    * the resulting df is uncategorized - start from the top!
###    * NOTE: verified that anti_join works on columns of list objects (author, editor)

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

### Load bibtex data
# message('Loading bibtex data...')
# bib_clean_fs <- list.files(path = here('_data/bibtex_clean'), 
#                            pattern = 'wos.bib|scopus.bib',
#                            full.names = TRUE)
#   
# screened_df <- bib2df(here('shiny_title_screen/app_out/title_screened.bib')) %>%
#   sample_n(5) %>%
#   mutate(EXTRA = 'some stuff')
# df2bib(screened_df, here('shiny_title_screen/app_out/title_screened.bib'))

### Define UI for application
ui <- fluidPage(

    ### Application title
    titlePanel("Shiny Title Screening"),

    ### Set up a tabset panel layout
    tabsetPanel(
      
      ### First panel: Welcome/introduction
      tabPanel(title = 'Welcome',
        ### Sidebar with a DT::datatable of the entire bib set?
        sidebarLayout(
          sidebarPanel(
            actionButton(
              inputId = 'load_bibtex',
              label = 'Load bibtex!'
            )
          ), ### end sidebar panel
          
          ### Show a preview of the loaded bibtex
          mainPanel(
            h2('Preview loaded bibtex:'),
            DTOutput('toscreen_table')
          ) ### end main panel
        )
      ), ### end Welcome tabPanel
      
      ### Second panel: perform the screening
      tabPanel(title = 'Screening',
               
        ### Sidebar with checkboxes for title screening criteria 
        sidebarLayout(
          sidebarPanel(
            ### Checkboxes for criteria:
            ### * Explicitly mentions satellite or earth obs in title
            ### * Explicitly mentions comparison context
            ### * Explicitly mentions a societal benefit
            checkboxGroupInput(
              inputId = 'criteria',
              label = 'Screening criteria:',
              choices = c('Satellite/EO data?'      = 'earth obs', 
                          'Comparison context?'     = 'comparison',
                          'Societal value/benefit?' = 'soc value')
              ), ### end of checkboxGroupInput
            
            ### Radio buttons for categorization
            ### * Definitely in scope
            ### * Maybe (keep)
            ### * Definitely out of scope
            ### Action button to store values and select new title
            radioButtons(
              inputId = 'screen_decision',
              label = 'Screening decision:',
              choices = c('In scope'        = 'keep',
                          'Likely (keep)'   = 'likely',
                          'Unlikely (keep)' = 'unlikely',
                          'Out of scope'    = 'omit')
              ), ### end of radio buttons
            ### * Append record to an output bibtex file with categorization in extra field
            actionButton(
              inputId = 'screen_act',
              label = 'Do it!'
            )
          ), ### end sidebar panel
      
          ### Show information on a selected title
          mainPanel(
            ### Display the title with search terms highlighted
            ### Display author(s), year, journal
            ### Display the abstract with search terms highlighted
          ) ### end main panel
          
        ) ### end sidebarLayout
      ) ### end tabPanel for screening
    ) ### end tabsetPanel layout
) ### end fluidPage ui

# Define server logic required to draw a histogram
server <- function(input, output) {

  ###################################
  ### Read and display raw bibtex ###
  ###################################

  ### Function to read bibtex given file selection (input$bibtex_fs) and action (input$load_bibtex)
  alldocs_df <- eventReactive(
    input$load_bibtex, {
      message('in alldocs_df() reactive')
      fs <- list.files(here('_data/bibtex_clean'),
                       pattern = 'zot_benchmark_a.bib',
                       # pattern = 'wos.bib|scopus.bib',
                       full.names = TRUE)
      df <- lapply(fs, bib2df::bib2df) %>%
        setNames(basename(fs)) %>%
        bind_rows(.id = 'bibtex_source') %>%
        distinct()
      return(df)
    }
  )
  toscreen_df <- reactive({
    message('in toscreen_df() reactive')
    screened <- bib2df::bib2df('app_out/title_screened.bib')

    toscreen <- alldocs_df() %>%
      anti_join(screened %>% select(-EXTRA))
    message('... data frame in toscreen_df() has ', nrow(toscreen), ' rows!')
    return(toscreen)
  })
  
  output$toscreen_table <- renderDataTable({
    df <- toscreen_df() %>%
      head(20) %>%
      select(TITLE, JOURNAL, YEAR, AUTHOR) %>%
      ### unspool authors and select the first author only
      unnest(AUTHOR) %>%
      group_by(TITLE, JOURNAL, YEAR) %>%
      slice(1) %>%
      ungroup() %>%
      rename(FIRST_AUTHOR = AUTHOR) %>%
      mutate(TITLE = str_remove_all(TITLE, '\\{|\\}'))
    
    DT::datatable(df)
  })
  
  ### Function to anti_join with existing screened output
  
  
  ### output to display screened output
  ### output to display unscreened output
  ### output to display selected doc for screening: highlight search terms in title and abstract

}

# Run the application 
shinyApp(ui = ui, server = server)
