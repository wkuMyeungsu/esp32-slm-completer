idf_component_register(SRCS "main.c" "model.c"
                       PRIV_REQUIRES esp_driver_usb_serial_jtag esp_timer esp_partition esp_app_format)
