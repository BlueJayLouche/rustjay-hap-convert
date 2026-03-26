mod app;
mod encode;
mod job;
mod probe;

fn main() -> eframe::Result<()> {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([720.0, 480.0])
            .with_min_inner_size([520.0, 320.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };

    eframe::run_native(
        "Rustjay Hap Converter",
        options,
        Box::new(|cc| Ok(Box::new(app::HapConvertApp::new(cc)))),
    )
}
