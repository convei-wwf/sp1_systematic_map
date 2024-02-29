
### Define UI for application
ui <- navbarPage(
  title = "Shiny Title Screening",
  theme = bslib::bs_theme(bootswatch = 'morph'),


  # ### First panel: Welcome/introduction
  tabPanel(title = 'Welcome',
    ### Sidebar with a DT::datatable of the entire bib set?
    sidebarLayout(
      sidebarPanel(
      #   shinyFilesButton(
      #     id = 'bib_file',
      #     label = 'choose bib file...',
      #     title = 'Select a bibliography file...',
      #     multiple = FALSE),
      #   
      #   shinyFilesButton(
      #     id = 'screened_file',
      #     label = 'choose screened file...',
      #     title = 'Select a file for screening results...',
      #     multiple = FALSE),
      #   
      #   actionButton(
      #     inputId = 'merge_bibs',
      #     label = 'Prep to screen', icon = icon('book')),
        
        radioButtons(
          inputId = 'df_preview',
          label = 'Preview:',
          choices = c('all', 'screened', 'to screen' = 'toscreen'),
          selected = 'all'),
      ), ### end sidebar panel

      ### Show a preview of the loaded bibtex
      mainPanel(
        DTOutput('toscreen_preview')
      ) ### end main panel
    )
  ), ### end Welcome tabPanel
  
  ### Second panel: perform the screening
  tabPanel(title = 'Screening',

    ### Sidebar with checkboxes for title screening criteria
    sidebarLayout(
      sidebarPanel(
        includeMarkdown('criteria.md'),

        ### Radio buttons for categorization
        radioButtons(
          inputId = 'screen_decision',
          label = 'Benchmark paper according to title?:',
          choices = c('Definitely in scope',
                      'Earth Observation context',
                      'Applied Science context',
                      'Not in scope'),
          selected = character(0)
          ), ### end of radio buttons
        ### * Append record to an output bibtex file with categorization in extra field
        actionButton(
          inputId = 'reveal_abstr',
          label = 'Reveal abstract?'
        ),
        actionButton(
          inputId = 'screen_action',
          label = 'Log it!'
        ),
        actionButton(
          inputId = 'skip_doc',
          label = 'Skip document!'
        )
      ), ### end sidebar panel

      ### Show information on a selected title
      mainPanel(
        htmlOutput('doc_fields_text')
      ) ### end main panel

    ) ### end sidebarLayout
  ) ### end tabPanel for screening

) ### end fluidPage ui

