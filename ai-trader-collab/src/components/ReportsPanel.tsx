import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { FileText, ExternalLink } from "lucide-react";
import { ReportData } from "./Dashboard";
import { useState } from "react";

interface ReportsPanelProps {
  reports: ReportData[];
}

export const ReportsPanel = ({ reports }: ReportsPanelProps) => {
  const [expandedReport, setExpandedReport] = useState<string | null>(null);
  if (reports.length === 0) {
    return (
      <div className="p-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Recent Reports
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-col items-center justify-center h-32 text-center">
              <FileText className="h-12 w-12 text-muted-foreground mb-4" />
              <p className="text-muted-foreground">Reports will appear after analysis</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="p-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Recent Reports
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {reports.map((report) => (
              <div key={report.id}>
                <div
                  onClick={() => setExpandedReport(expandedReport === report.id ? null : report.id)}
                  className="border border-border rounded-lg p-4 bg-background hover:border-primary/50 hover:shadow-lg transition-all cursor-pointer"
                >
                  <div className="mb-3">
                    <h4 className="font-semibold mb-1 flex items-center gap-2">
                      {report.title}
                      <ExternalLink className="h-3 w-3" />
                    </h4>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <span>{report.agentName}</span>
                      <span>•</span>
                      <span>{new Date(report.timestamp).toLocaleTimeString()}</span>
                    </div>
                  </div>

                  <p className="text-sm text-muted-foreground line-clamp-3">
                    {report.preview}
                  </p>
                </div>
                
                {expandedReport === report.id && (
                  <div className="mt-2 p-4 border border-primary rounded-lg bg-card max-h-96 overflow-y-auto">
                    <h5 className="font-semibold mb-3 text-primary">{report.title} - Full Report</h5>
                    <div className="text-sm whitespace-pre-line leading-relaxed">
                      {report.fullText || report.preview || "No content available"}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
