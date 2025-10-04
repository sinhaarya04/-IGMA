import { Button } from "./ui/button";
import { Progress } from "./ui/progress";
import { FileImage, FileText } from "lucide-react";

interface DashboardHeaderProps {
  status: "connected" | "connecting" | "disconnected";
  progress: number;
  isAnalyzing: boolean;
  onExportPNG: () => void;
  onExportPDF: () => void;
}

export const DashboardHeader = ({
  status,
  progress,
  isAnalyzing,
  onExportPNG,
  onExportPDF,
}: DashboardHeaderProps) => {
  const statusColor = {
    connected: "bg-status-connected",
    connecting: "bg-status-connecting",
    disconnected: "bg-status-disconnected",
  }[status];

  const statusText = {
    connected: "Connected",
    connecting: "Connecting...",
    disconnected: "Disconnected",
  }[status];

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/60">
      <div className="flex items-center justify-between p-4">
        <div className="flex items-center gap-4">
          <h1 className="text-2xl font-bold text-primary flex items-center gap-2">
            <span className="text-3xl">📊</span>
            ΣIGMA Dashboard
          </h1>
          <div className="flex items-center gap-2">
            <div className={`h-2 w-2 rounded-full ${statusColor} animate-pulse`} />
            <span className="text-sm text-muted-foreground">{statusText}</span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={onExportPNG}
            disabled={!isAnalyzing && progress === 0}
          >
            <FileImage className="h-4 w-4 mr-2" />
            Export PNG
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={onExportPDF}
            disabled={!isAnalyzing && progress === 0}
          >
            <FileText className="h-4 w-4 mr-2" />
            Export PDF
          </Button>
        </div>
      </div>

      {isAnalyzing && (
        <div className="px-4 pb-4">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs text-muted-foreground">Progress:</span>
            <span className="text-xs font-medium">{progress}%</span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>
      )}
    </header>
  );
};
