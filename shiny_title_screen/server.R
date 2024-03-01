
server <- shinyServer(function(input, output) {

  observeEvent(input$bib_file, {
    message('observeEvent triggered')
    shinyFileChoose(input, 'bib_file', roots=c(wd = here::here()), filetypes=c('', 'txt'),
                    defaultPath='', defaultRoot='wd')
  })
})
